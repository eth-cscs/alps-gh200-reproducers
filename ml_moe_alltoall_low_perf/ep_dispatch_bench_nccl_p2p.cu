/*
 * GPU EP dispatch/combine benchmark using NCCL.
 *
 * Mirrors ep_dispatch_bench_mpi_cpu_p2p.c but places buffers on the GPU and
 * uses ncclAllToAll for dispatch/combine and ncclSend/ncclRecv for pairwise
 * point-to-point measurements.
 *
 * Build:
 *     nvcc -O2 -std=c++11 -ccbin mpicxx \
 *          -I${NCCL_DIR}/include -L${NCCL_DIR}/lib \
 *          -lnccl -o ep_dispatch_bench_nccl_p2p ep_dispatch_bench_nccl_p2p.cu
 *
 * Launch:
 *     srun --mpi=pmix --ntasks-per-node=4 \
 *          ./ep_dispatch_bench_nccl_p2p --ep 8 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <nccl.h>

static long getenv_long(const char *name, long def) {
    const char *v = getenv(name);
    if (!v) return def;
    char *end;
    long r = strtol(v, &end, 10);
    if (end == v || *end) return def;
    return r;
}

static int local_rank_from_env(void) {
    int lr = (int)getenv_long("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (lr >= 0) return lr;
    lr = (int)getenv_long("SLURM_LOCALID", -1);
    if (lr >= 0) return lr;
    return 0;
}

/* xoshiro128** PRNG */
static uint32_t xoshiro128ss(uint64_t *s) {
    uint64_t x = s[0];
    uint64_t y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return (uint32_t)(s[1] + y);
}

static uint16_t random_u16(uint64_t *s) {
    return (uint16_t)(xoshiro128ss(s) & 0xFFFFU);
}

static uint32_t random_u32(uint64_t *s) {
    return xoshiro128ss(s);
}

static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const char *p = s; p && *p; ++p) {
        switch (*p) {
            case '"': fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\b': fputs("\\b", f); break;
            case '\f': fputs("\\f", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if ((unsigned char)*p < 0x20)
                    fprintf(f, "\\u%04x", (unsigned char)*p);
                else
                    fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void write_float_array(FILE *f, const char *name, const double *arr, int n) {
    fprintf(f, "    \"%s\": [", name);
    for (int i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        fprintf(f, "%.3f", arr[i]);
    }
    fprintf(f, "]");
}

static void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

static void nccl_check(ncclResult_t r, const char *what) {
    if (r != ncclSuccess) {
        fprintf(stderr, "NCCL error in %s: %s\n", what, ncclGetErrorString(r));
        exit(1);
    }
}

static double median_double(const double *arr, int n) {
    if (n <= 0) return 0.0;
    double *tmp = (double *)malloc(n * sizeof(double));
    memcpy(tmp, arr, n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (tmp[j] < tmp[i]) { double t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
        }
    }
    double r = (n % 2) ? tmp[n/2] : (tmp[n/2 - 1] + tmp[n/2]) / 2.0;
    free(tmp);
    return r;
}

static void gather_affinity(int local_rank, char *model, size_t sz) {
    (void)local_rank;
    model[0] = '\0';
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) {
        snprintf(model, sz, "%s", prop.name);
    }
}

/* Globals for dispatch/combine helpers. */
static uint16_t *g_send;
static uint16_t *g_recv;
static uint16_t *g_cmb_send;
static uint16_t *g_cmb_recv;
static int g_count;
static int g_world;
static ncclComm_t g_comm;
static cudaStream_t g_stream;

/* Emulate ncclAllToAll using ncclSend/ncclRecv so this works with older NCCL
 * versions that do not export ncclAllToAll.  Each rank sends slice r to peer r
 * and receives slice r from peer r. */
static void do_nccl_alltoall(const uint16_t *sendbuf, uint16_t *recvbuf) {
    nccl_check(ncclGroupStart(), "group start alltoall");
    for (int r = 0; r < g_world; ++r) {
        nccl_check(ncclSend(sendbuf + (size_t)r * g_count, (size_t)g_count, ncclBfloat16, r, g_comm, g_stream),
                   "ncclSend alltoall");
        nccl_check(ncclRecv(recvbuf + (size_t)r * g_count, (size_t)g_count, ncclBfloat16, r, g_comm, g_stream),
                   "ncclRecv alltoall");
    }
    nccl_check(ncclGroupEnd(), "group end alltoall");
}

static void do_dispatch(void) {
    do_nccl_alltoall(g_send, g_recv);
}

static void do_combine(void) {
    do_nccl_alltoall(g_cmb_send, g_cmb_recv);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    g_world = world;

    /* Dump environment variables to stdout for reproducibility/debugging. */
    if (rank == 0) {
        printf("=== environment after MPI_Init (rank 0 of %d) ===\n", world);
        extern char **environ;
        const char *prefixes[] = {
            "FI_", "MPICH_", "OMPI_MCA_", "OPAL_", "PMIX_", "MPI_",
            "CXI_", "GATHER_CXI", "NUMA", "OMP_", "SLURM_", "LDMS_",
            "FI_CXI", "NCCL_", "CUDA_", NULL
        };
        for (char **e = environ; *e; ++e) {
            for (int i = 0; prefixes[i]; ++i) {
                if (strncmp(*e, prefixes[i], strlen(prefixes[i])) == 0) {
                    printf("%s\n", *e);
                    break;
                }
            }
        }
        printf("=== end environment ===\n");
        fflush(stdout);
    }

    int num_tokens = 8192;
    int hidden = 1792;
    int num_topk = 8;
    int num_experts = 256;
    int ep = world;
    int iters = 200;
    int warmup = 10;
    int seed = 0;
    int chunk_id = 0;
    int total_chunks = 1;
    int no_combine = 0;
    int no_p2p = 0;
    const char *outdir = "results-ep-nccl";
    const char *outfile = NULL;
    const char *pin = "";

    static struct option longopts[] = {
        {"num-tokens", required_argument, 0, 0},
        {"hidden", required_argument, 0, 0},
        {"num-topk", required_argument, 0, 0},
        {"num-experts", required_argument, 0, 0},
        {"ep", required_argument, 0, 0},
        {"iters", required_argument, 0, 0},
        {"warmup", required_argument, 0, 0},
        {"seed", required_argument, 0, 0},
        {"chunk-id", required_argument, 0, 0},
        {"total-chunks", required_argument, 0, 0},
        {"no-combine", no_argument, 0, 0},
        {"no-p2p", no_argument, 0, 0},
        {"outdir", required_argument, 0, 0},
        {"out", required_argument, 0, 0},
        {"pin", required_argument, 0, 0},
        {"balanced-routing", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c;
    int optidx;
    while ((c = getopt_long(argc, argv, "", longopts, &optidx)) != -1) {
        if (c != 0) continue;
        const char *name = longopts[optidx].name;
        const char *val = optarg ? optarg : "";
        if (!strcmp(name, "num-tokens")) num_tokens = atoi(val);
        else if (!strcmp(name, "hidden")) hidden = atoi(val);
        else if (!strcmp(name, "num-topk")) num_topk = atoi(val);
        else if (!strcmp(name, "num-experts")) num_experts = atoi(val);
        else if (!strcmp(name, "ep")) ep = atoi(val);
        else if (!strcmp(name, "iters")) iters = atoi(val);
        else if (!strcmp(name, "warmup")) warmup = atoi(val);
        else if (!strcmp(name, "seed")) seed = atoi(val);
        else if (!strcmp(name, "chunk-id")) chunk_id = atoi(val);
        else if (!strcmp(name, "total-chunks")) total_chunks = atoi(val);
        else if (!strcmp(name, "no-combine")) no_combine = 1;
        else if (!strcmp(name, "no-p2p")) no_p2p = 1;
        else if (!strcmp(name, "outdir")) outdir = val;
        else if (!strcmp(name, "out")) outfile = val;
        else if (!strcmp(name, "pin")) pin = val;
        else if (!strcmp(name, "balanced-routing")) { /* ignored, always balanced */ }
    }

    if (world % ep != 0) {
        if (rank == 0) fprintf(stderr, "WORLD_SIZE=%d is not a multiple of --ep %d\n", world, ep);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = local_rank_from_env();

    printf("[rank %d] local_rank=%d  starting GPU setup\n", rank, local_rank);
    fflush(stdout);

    /* Select GPU. */
    int dev_count;
    cuda_check(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount");
    int dev_id = local_rank % dev_count;
    printf("[rank %d] selecting GPU %d of %d\n", rank, dev_id, dev_count);
    fflush(stdout);
    cuda_check(cudaSetDevice(dev_id), "cudaSetDevice");
    cuda_check(cudaStreamCreate(&g_stream), "cudaStreamCreate");

    /* Init NCCL. */
    printf("[rank %d] initializing NCCL comm for world=%d\n", rank, world);
    fflush(stdout);
    ncclUniqueId id;
    if (rank == 0) nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    nccl_check(ncclCommInitRank(&g_comm, world, id, rank), "ncclCommInitRank");
    printf("[rank %d] NCCL comm initialized\n", rank);
    fflush(stdout);

    if (num_tokens * num_topk % ep != 0) {
        if (rank == 0) fprintf(stderr, "num_tokens*num_topk not divisible by ep\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int experts_per_rank = num_experts / ep;
    int count_per_rank = num_tokens * num_topk / ep;

    /* Balanced topk generation on CPU. */
    int total_topk = num_tokens * num_topk;
    int *topk_idx = (int *)malloc(total_topk * sizeof(int));
    uint64_t rng_state[2] = { (uint64_t)(seed + rank + 1), (uint64_t)(seed + rank + 7777) };
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < num_topk; ++k) {
            int r = (t + k) % ep;
            int expert = (int)(random_u32(rng_state) % (uint32_t)experts_per_rank);
            topk_idx[t * num_topk + k] = r * experts_per_rank + expert;
        }
    }

    /* Compute permutation. */
    int **rank_slots = (int **)calloc(ep, sizeof(int *));
    for (int r = 0; r < ep; ++r) rank_slots[r] = (int *)malloc(count_per_rank * sizeof(int));
    int *send_counts = (int *)calloc(ep, sizeof(int));
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < num_topk; ++k) {
            int eidx = topk_idx[t * num_topk + k];
            int r = eidx / experts_per_rank;
            if (send_counts[r] >= count_per_rank) {
                fprintf(stderr, "[rank %d] rank %d send count overflow\n", rank, r);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            rank_slots[r][send_counts[r]++] = t;
        }
    }

    /* Allocate GPU send/recv buffers. */
    size_t slice_elems = (size_t)count_per_rank * hidden;
    size_t total_elems = slice_elems * ep;
    size_t total_bytes = total_elems * sizeof(uint16_t);
    printf("[rank %d] allocating GPU buffers: total_bytes=%zu\n", rank, total_bytes);
    fflush(stdout);
    cuda_check(cudaMalloc(&g_send, total_bytes), "cudaMalloc g_send");
    cuda_check(cudaMalloc(&g_recv, total_bytes), "cudaMalloc g_recv");
    cuda_check(cudaMalloc(&g_cmb_send, total_bytes), "cudaMalloc g_cmb_send");
    cuda_check(cudaMalloc(&g_cmb_recv, total_bytes), "cudaMalloc g_cmb_recv");
    printf("[rank %d] GPU buffers allocated\n", rank);
    fflush(stdout);

    /* Generate input x and permute into send buffer on host, then copy to GPU. */
    uint16_t *x = (uint16_t *)malloc((size_t)num_tokens * hidden * sizeof(uint16_t));
    uint16_t *send_host = (uint16_t *)malloc(total_bytes);
    uint16_t *cmb_send_host = (uint16_t *)malloc(total_bytes);
    for (size_t i = 0; i < (size_t)num_tokens * hidden; ++i) x[i] = random_u16(rng_state);
    for (int r = 0; r < ep; ++r) {
        for (int i = 0; i < count_per_rank; ++i) {
            int t = rank_slots[r][i];
            memcpy(send_host + ((size_t)r * count_per_rank + i) * hidden,
                   x + (size_t)t * hidden, hidden * sizeof(uint16_t));
        }
    }
    memset(cmb_send_host, 0, total_bytes);
    cuda_check(cudaMemcpy(g_send, send_host, total_bytes, cudaMemcpyHostToDevice), "H2D send");
    cuda_check(cudaMemcpy(g_cmb_send, cmb_send_host, total_bytes, cudaMemcpyHostToDevice), "H2D cmb_send");
    free(x); free(send_host); free(cmb_send_host);

    g_count = (int)slice_elems;
    if ((long)count_per_rank * hidden > INT_MAX) {
        if (rank == 0) fprintf(stderr, "NCCL count overflow\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("[rank %d] g_count=%d (per-peer element count)\n", rank, g_count);
    fflush(stdout);

    /* Timing events. */
    cudaEvent_t ev_start, ev_stop;
    cuda_check(cudaEventCreate(&ev_start), "event create start");
    cuda_check(cudaEventCreate(&ev_stop), "event create stop");

    auto measure_us = [&](void (*fn)(void), double *out, int n) {
        for (int i = 0; i < n; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            cuda_check(cudaEventRecord(ev_start, g_stream), "event record start");
            fn();
            MPI_Barrier(MPI_COMM_WORLD);
            cuda_check(cudaEventRecord(ev_stop, g_stream), "event record stop");
            cuda_check(cudaStreamSynchronize(g_stream), "stream sync");
            float ms;
            cuda_check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "event elapsed");
            out[i] = ms * 1e3;
        }
    };

    /* Sanity dispatch. */
    printf("[rank %d] issuing first dispatch (this triggers NCCL connection setup)\n", rank);
    fflush(stdout);
    do_dispatch();
    printf("[rank %d] first dispatch issued, syncing stream...\n", rank);
    fflush(stdout);
    cuda_check(cudaStreamSynchronize(g_stream), "stream sync");
    printf("[rank %d] first dispatch stream sync done\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    double *dispatch_us = (double *)malloc(iters * sizeof(double));
    double *combine_us = (double *)malloc(iters * sizeof(double));

    for (int i = 0; i < warmup; ++i) do_dispatch();
    cuda_check(cudaStreamSynchronize(g_stream), "stream sync");
    MPI_Barrier(MPI_COMM_WORLD);

    printf("[rank %d] starting timed dispatch iterations\n", rank);
    fflush(stdout);
    measure_us(do_dispatch, dispatch_us, iters);
    printf("[rank %d] timed dispatch iterations done\n", rank);
    fflush(stdout);

    if (!no_combine) {
        printf("[rank %d] starting combine phase\n", rank);
        fflush(stdout);
        for (int i = 0; i < warmup; ++i) do_combine();
        cuda_check(cudaStreamSynchronize(g_stream), "stream sync");
        MPI_Barrier(MPI_COMM_WORLD);

        measure_us(do_combine, combine_us, iters);
        printf("[rank %d] combine phase done\n", rank);
        fflush(stdout);
    } else {
        for (int i = 0; i < iters; ++i) combine_us[i] = 0.0;
    }

    /* Point-to-point pairwise benchmark using NCCL send/recv. */
    int p2p_warmup = 5;
    int p2p_iters = 200;
    size_t p2p_bytes = (size_t)g_count * sizeof(uint16_t);
    uint8_t *d_p2p_send = NULL;
    uint8_t *d_p2p_recv = NULL;
    double **p2p_us = NULL;
    int *p2p_peer_valid = NULL;

    if (!no_p2p) {
        printf("[rank %d] starting p2p benchmark, p2p_bytes=%zu\n", rank, p2p_bytes);
        fflush(stdout);
        cuda_check(cudaMalloc(&d_p2p_send, p2p_bytes), "cudaMalloc p2p_send");
        cuda_check(cudaMalloc(&d_p2p_recv, p2p_bytes), "cudaMalloc p2p_recv");
        {
            uint8_t *h = (uint8_t *)malloc(p2p_bytes);
            memset(h, 0xAB, p2p_bytes);
            cuda_check(cudaMemcpy(d_p2p_send, h, p2p_bytes, cudaMemcpyHostToDevice), "H2D p2p_send");
            free(h);
        }

        p2p_us = (double **)calloc(world, sizeof(double *));
        p2p_peer_valid = (int *)calloc(world, sizeof(int));
        for (int d = 0; d < world; ++d) {
            if (d == rank) continue;
            p2p_us[d] = (double *)malloc(p2p_iters * sizeof(double));
            p2p_peer_valid[d] = 1;
        }

        for (int src = 0; src < world; ++src) {
            for (int dst = 0; dst < world; ++dst) {
                if (src == dst) continue;
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == src || rank == dst) {
                    int peer = (rank == src) ? dst : src;
                    if (rank == src) {
                        printf("[rank %d] p2p pair src=%d dst=%d peer=%d warmup start\n", rank, src, dst, peer);
                        fflush(stdout);
                    }
                    for (int w = 0; w < p2p_warmup; ++w) {
                        nccl_check(ncclGroupStart(), "group start warmup");
                        if (rank == src) {
                            nccl_check(ncclRecv(d_p2p_recv, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "recv warmup");
                            nccl_check(ncclSend(d_p2p_send, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "send warmup");
                        } else {
                            nccl_check(ncclRecv(d_p2p_recv, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "recv warmup");
                            nccl_check(ncclSend(d_p2p_send, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "send warmup");
                        }
                        nccl_check(ncclGroupEnd(), "group end warmup");
                        cuda_check(cudaStreamSynchronize(g_stream), "stream sync warmup");
                    }
                    if (rank == src) {
                        printf("[rank %d] p2p pair src=%d dst=%d warmup done, timed iters start\n", rank, src, dst);
                        fflush(stdout);
                    }
                    for (int i = 0; i < p2p_iters; ++i) {
                        cuda_check(cudaEventRecord(ev_start, g_stream), "event record p2p start");
                        nccl_check(ncclGroupStart(), "group start p2p");
                        if (rank == src) {
                            nccl_check(ncclRecv(d_p2p_recv, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "recv p2p");
                            nccl_check(ncclSend(d_p2p_send, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "send p2p");
                        } else {
                            nccl_check(ncclRecv(d_p2p_recv, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "recv p2p");
                            nccl_check(ncclSend(d_p2p_send, p2p_bytes, ncclUint8, peer, g_comm, g_stream), "send p2p");
                        }
                        nccl_check(ncclGroupEnd(), "group end p2p");
                        cuda_check(cudaStreamSynchronize(g_stream), "stream sync p2p");
                        cuda_check(cudaEventRecord(ev_stop, g_stream), "event record p2p stop");
                        cuda_check(cudaEventSynchronize(ev_stop), "event sync p2p stop");
                        float ms;
                        cuda_check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "event elapsed p2p");
                        if (rank == src) p2p_us[dst][i] = ms * 1e3;
                    }
                    if (rank == src) {
                        printf("[rank %d] p2p pair src=%d dst=%d done\n", rank, src, dst);
                        fflush(stdout);
                    }
                }
            }
        }
        printf("[rank %d] p2p benchmark complete\n", rank);
        fflush(stdout);
    }

    cuda_check(cudaEventDestroy(ev_start), "event destroy start");
    cuda_check(cudaEventDestroy(ev_stop), "event destroy stop");

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hlen;
    MPI_Get_processor_name(hostname, &hlen);

    char gpu_model[256];
    gather_affinity(local_rank, gpu_model, sizeof(gpu_model));

    /* Aggregate hostnames at rank 0. */
    int names_len = (int)strlen(hostname) + 1;
    int *all_name_lens = NULL;
    char *all_names = NULL;
    if (rank == 0) all_name_lens = (int *)malloc(world * sizeof(int));
    MPI_Gather(&names_len, 1, MPI_INT, all_name_lens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < world; ++i) total += all_name_lens[i];
        all_names = (char *)malloc(total);
        int *displs = (int *)malloc(world * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < world; ++i) displs[i] = displs[i - 1] + all_name_lens[i - 1];
        MPI_Gatherv(hostname, names_len, MPI_CHAR, all_names, all_name_lens, displs, MPI_CHAR, 0, MPI_COMM_WORLD);
        free(displs);
    } else {
        MPI_Gatherv(hostname, names_len, MPI_CHAR, NULL, NULL, NULL, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    /* mkdir -p outdir (rank 0 only). */
    if (rank == 0) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", outdir);
        int st = system(cmd);
        (void)st;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Write per-rank JSON. */
    char per_rank_path[512];
    snprintf(per_rank_path, sizeof(per_rank_path), "%s/ep_custom_chunk%d_rank%d.json", outdir, chunk_id, rank);
    FILE *jf = fopen(per_rank_path, "w");
    if (!jf) {
        fprintf(stderr, "[rank %d] cannot open %s\n", rank, per_rank_path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(jf, "{\n");
    fprintf(jf, "  \"config\": {\n");
    fprintf(jf, "    \"num_tokens\": %d,\n", num_tokens);
    fprintf(jf, "    \"hidden\": %d,\n", hidden);
    fprintf(jf, "    \"num_topk\": %d,\n", num_topk);
    fprintf(jf, "    \"num_experts\": %d,\n", num_experts);
    fprintf(jf, "    \"ep\": %d,\n", ep);
    fprintf(jf, "    \"iters\": %d,\n", iters);
    fprintf(jf, "    \"warmup\": %d,\n", warmup);
    fprintf(jf, "    \"seed\": %d,\n", seed);
    fprintf(jf, "    \"balanced_routing\": true,\n");
    fprintf(jf, "    \"no_combine\": %s,\n", no_combine ? "true" : "false");
    fprintf(jf, "    \"chunk_id\": %d,\n", chunk_id);
    fprintf(jf, "    \"total_chunks\": %d,\n", total_chunks);
    fprintf(jf, "    \"pin\": \"%s\",\n", pin);
    fprintf(jf, "    \"outdir\": \"%s\"\n", outdir);
    fprintf(jf, "  },\n");
    fprintf(jf, "  \"rank\": %d,\n", rank);
    fprintf(jf, "  \"host\": \"%s\",\n", hostname);
    fprintf(jf, "  \"local_rank\": %d,\n", local_rank);
    fprintf(jf, "  \"gpu_id\": %d,\n", dev_id);
    fprintf(jf, "  \"group_id\": 0,\n");
    fprintf(jf, "  \"world_size\": %d,\n", world);
    fprintf(jf, "  \"ep\": %d,\n", ep);
    fprintf(jf, "  \"chunk_id\": %d,\n", chunk_id);
    fprintf(jf, "  \"total_chunks\": %d,\n", total_chunks);
    fprintf(jf, "  \"pin\": {\"mode\": \"%s\", \"requested\": %s},\n", pin, pin[0] ? "true" : "false");
    fprintf(jf, "  \"affinity\": {\"backend\": \"nccl\", \"local_rank\": %d, \"gpu_model\": ", local_rank);
    json_string(jf, gpu_model);
    fprintf(jf, "},\n");
    fprintf(jf, "  \"gpu_health\": {},\n");
    fprintf(jf, "  \"per_rank_tokens\": [");
    for (int i = 0; i < ep; ++i) {
        if (i) fputc(',', jf);
        fprintf(jf, "%d", count_per_rank);
    }
    fprintf(jf, "],\n");
    fprintf(jf, "  \"per_node_tokens\": [");
    int ranks_per_node = (int)getenv_long("SLURM_NTASKS_PER_NODE", 1);
    for (int i = 0; i < ep; i += ranks_per_node) {
        if (i) fputc(',', jf);
        int sum = 0;
        for (int j = i; j < i + ranks_per_node && j < ep; ++j) sum += count_per_rank;
        fprintf(jf, "%d", sum);
    }
    fprintf(jf, "],\n");
    write_float_array(jf, "dispatch_us", dispatch_us, iters);
    fprintf(jf, ",\n");
    write_float_array(jf, "combine_us", combine_us, iters);
    fprintf(jf, ",\n");
    fprintf(jf, "  \"p2p_enabled\": %s,\n", no_p2p ? "false" : "true");
    if (!no_p2p) {
        fprintf(jf, "  \"p2p_pair_bytes\": %zu,\n", p2p_bytes);
        fprintf(jf, "  \"p2p_warmup\": %d,\n", p2p_warmup);
        fprintf(jf, "  \"p2p_iters\": %d,\n", p2p_iters);
        fprintf(jf, "  \"p2p_pairs\": [\n");
        bool first_p2p = true;
        for (int d = 0; d < world; ++d) {
            if (!p2p_peer_valid[d]) continue;
            if (!first_p2p) fprintf(jf, ",\n");
            first_p2p = false;
            fprintf(jf, "    {\"dst\": %d, \"us\": [", d);
            for (int i = 0; i < p2p_iters; ++i) {
                if (i) fputc(',', jf);
                fprintf(jf, "%.3f", p2p_us[d][i]);
            }
            fprintf(jf, "]}");
        }
        fprintf(jf, "\n  ],\n");
    }
    fprintf(jf, "  \"mpi_cpu_staging\": false\n");
    fprintf(jf, "}\n");
    fflush(jf);
    fsync(fileno(jf));
    fclose(jf);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Aggregate p2p summary across all ranks. */
    int pairs = no_p2p ? 0 : world * (world - 1);
    double *all_min = NULL;
    double *all_max = NULL;
    double *all_med = NULL;
    int    *all_src = NULL;
    int    *all_dst = NULL;
    if (pairs > 0) {
        all_min = (double *)malloc(pairs * sizeof(double));
        all_max = (double *)malloc(pairs * sizeof(double));
        all_med = (double *)malloc(pairs * sizeof(double));
        all_src = (int *)malloc(pairs * sizeof(int));
        all_dst = (int *)malloc(pairs * sizeof(int));
    }
    int pair_idx = 0;
    for (int s = 0; s < world; ++s) {
        for (int d = 0; d < world; ++d) {
            if (s == d) continue;
            double sample[1] = {0.0};
            if (!no_p2p && rank == s) {
                memcpy(sample, &p2p_us[d][0], sizeof(double));
            }
            MPI_Bcast(sample, 1, MPI_DOUBLE, s, MPI_COMM_WORLD);
            double min_us = sample[0];
            double max_us = sample[0];
            double med_us = sample[0];
            if (!no_p2p && rank == s) {
                min_us = p2p_us[d][0];
                max_us = p2p_us[d][0];
                for (int i = 1; i < p2p_iters; ++i) {
                    if (p2p_us[d][i] < min_us) min_us = p2p_us[d][i];
                    if (p2p_us[d][i] > max_us) max_us = p2p_us[d][i];
                }
                med_us = median_double(p2p_us[d], p2p_iters);
            }
            double stats[3] = {min_us, max_us, med_us};
            MPI_Bcast(stats, 3, MPI_DOUBLE, s, MPI_COMM_WORLD);
            if (pairs > 0) {
                all_src[pair_idx] = s;
                all_dst[pair_idx] = d;
                all_min[pair_idx] = stats[0];
                all_max[pair_idx] = stats[1];
                all_med[pair_idx] = stats[2];
            }
            pair_idx++;
        }
    }

    /* Aggregate output at rank 0. */
    if (rank == 0 && outfile) {
        FILE *of = fopen(outfile, "w");
        if (of) {
            fprintf(of, "{\n");
            fprintf(of, "  \"config\": {\n");
            fprintf(of, "    \"num_tokens\": %d,\n", num_tokens);
            fprintf(of, "    \"hidden\": %d,\n", hidden);
            fprintf(of, "    \"num_topk\": %d,\n", num_topk);
            fprintf(of, "    \"num_experts\": %d,\n", num_experts);
            fprintf(of, "    \"ep\": %d,\n", ep);
            fprintf(of, "    \"iters\": %d,\n", iters);
            fprintf(of, "    \"warmup\": %d,\n", warmup);
            fprintf(of, "    \"seed\": %d,\n", seed);
            fprintf(of, "    \"balanced_routing\": true,\n");
            fprintf(of, "    \"no_combine\": %s,\n", no_combine ? "true" : "false");
            fprintf(of, "    \"chunk_id\": %d,\n", chunk_id);
            fprintf(of, "    \"total_chunks\": %d,\n", total_chunks);
            fprintf(of, "    \"pin\": \"%s\",\n", pin);
            fprintf(of, "    \"outdir\": \"%s\"\n", outdir);
            fprintf(of, "  },\n");
            fprintf(of, "  \"ranks\": [\n");
            for (int i = 0; i < world; ++i) {
                char path[512];
                snprintf(path, sizeof(path), "%s/ep_custom_chunk%d_rank%d.json", outdir, chunk_id, i);
                FILE *rf = fopen(path, "r");
                if (rf) {
                    if (i) fprintf(of, ",\n");
                    char buf[4096];
                    size_t n;
                    while ((n = fread(buf, 1, sizeof(buf), rf)) > 0) fwrite(buf, 1, n, of);
                    fclose(rf);
                }
            }
            fprintf(of, "\n  ],\n");

            if (pairs > 0) {
                fprintf(of, "  \"p2p_summary\": {\n");
                fprintf(of, "    \"pair_bytes\": %zu,\n", p2p_bytes);
                fprintf(of, "    \"pairs\": [\n");
                for (int i = 0; i < pairs; ++i) {
                    fprintf(of, "      {\"src\": %d, \"dst\": %d, \"min_us\": %.3f, \"max_us\": %.3f, \"median_us\": %.3f}%s\n",
                            all_src[i], all_dst[i], all_min[i], all_max[i], all_med[i],
                            (i + 1 < pairs) ? "," : "");
                }
                fprintf(of, "    ]\n");
                fprintf(of, "  },\n");
            }

            fprintf(of, "  \"nodes\": [");
            bool first = true;
            for (int i = 0; i < world; ++i) {
                char *name = all_names;
                for (int j = 0; j < i; ++j) name += all_name_lens[j];
                bool dup = false;
                for (int j = 0; j < i; ++j) {
                    char *nj = all_names;
                    for (int k = 0; k < j; ++k) nj += all_name_lens[k];
                    if (strcmp(name, nj) == 0) { dup = true; break; }
                }
                if (!dup) {
                    if (!first) fprintf(of, ", ");
                    json_string(of, name);
                    first = false;
                }
            }
            fprintf(of, "]\n}\n");
            fclose(of);
            printf("WROTE %s\n", outfile);
        }
    }

    free(all_min); free(all_max); free(all_med); free(all_src); free(all_dst);

    if (d_p2p_send) cudaFree(d_p2p_send);
    if (d_p2p_recv) cudaFree(d_p2p_recv);
    if (p2p_us) {
        for (int d = 0; d < world; ++d) free(p2p_us[d]);
        free(p2p_us);
    }
    free(p2p_peer_valid);

    cudaFree(g_send); cudaFree(g_recv); cudaFree(g_cmb_send); cudaFree(g_cmb_recv);
    free(dispatch_us); free(combine_us);
    free(topk_idx); free(send_counts);
    for (int r = 0; r < ep; ++r) free(rank_slots[r]);
    free(rank_slots);
    free(all_name_lens); free(all_names);

    ncclCommDestroy(g_comm);
    cudaStreamDestroy(g_stream);
    MPI_Finalize();
    return 0;
}
