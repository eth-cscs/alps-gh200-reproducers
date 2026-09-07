/*
 * CPU-only EP dispatch/combine benchmark using MPI.
 *
 * Strips out CUDA entirely to isolate whether the node-assignment latency
 * variability is in the fabric/CPU path or tied to GPU staging.
 *
 * Build:
 *     mpicc -O2 -std=gnu11 -o ep_dispatch_bench_mpi_cpu ep_dispatch_bench_mpi_cpu.c
 *
 * Launch:
 *     srun --mpi=pmix --ntasks-per-node=1 \
 *          ./ep_dispatch_bench_mpi_cpu --ep 16 ...
 */

#define _GNU_SOURCE
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

#ifndef MPI_BFLOAT16_T
  #define USE_BFLOAT16_AS_UINT16 1
#endif

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

static void gather_affinity(int local_rank, char *cpuinfo_model, size_t model_size) {
    (void)local_rank;
    cpuinfo_model[0] = '\0';
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *p = strchr(line, ':');
            if (p) {
                p++;
                while (*p == ' ' || *p == '\t') p++;
                strncpy(cpuinfo_model, p, model_size - 1);
                cpuinfo_model[model_size - 1] = '\0';
                char *e = cpuinfo_model + strlen(cpuinfo_model) - 1;
                while (e >= cpuinfo_model && (*e == '\n' || *e == '\r')) *e-- = '\0';
            }
            break;
        }
    }
    fclose(f);
}

static double median_double(const double *arr, int n) {
    if (n <= 0) return 0.0;
    double *tmp = malloc(n * sizeof(double));
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

/* Globals for dispatch/combine helpers. */
static uint16_t *g_send;
static uint16_t *g_recv;
static uint16_t *g_cmb_send;
static uint16_t *g_cmb_recv;
static size_t g_total_bytes;
static int g_mpi_count;
static MPI_Datatype g_elem_type;

static void do_dispatch(void) {
    MPI_Alltoall(g_send, g_mpi_count, g_elem_type,
                 g_recv, g_mpi_count, g_elem_type, MPI_COMM_WORLD);
}

static void do_combine(void) {
    MPI_Alltoall(g_cmb_send, g_mpi_count, g_elem_type,
                 g_cmb_recv, g_mpi_count, g_elem_type, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    /* Dump environment variables to stdout for reproducibility/debugging.
     * Filter to variables that are likely to influence MPI/libfabric behavior. */
    if (rank == 0) {
        printf("=== environment after MPI_Init (rank 0 of %d) ===\n", world);
        extern char **environ;
        const char *prefixes[] = {
            "FI_", "MPICH_", "OMPI_MCA_", "OPAL_", "PMIX_", "MPI_",
            "CXI_", "GATHER_CXI", "NUMA", "OMP_", "SLURM_", "LDMS_",
            "FI_CXI", NULL
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
    int ep = 8;
    int iters = 200;
    int warmup = 10;
    int seed = 0;
    int chunk_id = 0;
    int total_chunks = 1;
    int no_combine = 0;
    int no_p2p = 0;
    const char *outdir = "results-ep-mpi-cpu";
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
    (void)local_rank;

    if (num_tokens * num_topk % ep != 0) {
        if (rank == 0) fprintf(stderr, "num_tokens*num_topk not divisible by ep\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int experts_per_rank = num_experts / ep;
    int count_per_rank = num_tokens * num_topk / ep;

    /* Balanced topk generation on CPU. */
    int total_topk = num_tokens * num_topk;
    int *topk_idx = malloc(total_topk * sizeof(int));
    uint64_t rng_state[2] = { (uint64_t)(seed + rank + 1), (uint64_t)(seed + rank + 7777) };
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < num_topk; ++k) {
            int r = (t + k) % ep;
            int expert = (int)(random_u32(rng_state) % (uint32_t)experts_per_rank);
            topk_idx[t * num_topk + k] = r * experts_per_rank + expert;
        }
    }

    /* Compute permutation. */
    int **rank_slots = calloc(ep, sizeof(int *));
    for (int r = 0; r < ep; ++r) rank_slots[r] = malloc(count_per_rank * sizeof(int));
    int *send_counts = calloc(ep, sizeof(int));
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

    /* Allocate CPU send/recv buffers. */
    size_t slice_elems = (size_t)count_per_rank * hidden;
    size_t total_elems = slice_elems * ep;
    size_t total_bytes = total_elems * sizeof(uint16_t);
    uint16_t *send = malloc(total_bytes);
    uint16_t *recv = malloc(total_bytes);
    uint16_t *cmb_send = malloc(total_bytes);
    uint16_t *cmb_recv = malloc(total_bytes);
    if (!send || !recv || !cmb_send || !cmb_recv) {
        fprintf(stderr, "[rank %d] malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Generate input x and permute into send buffer. */
    uint16_t *x = malloc((size_t)num_tokens * hidden * sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)num_tokens * hidden; ++i) x[i] = random_u16(rng_state);
    for (int r = 0; r < ep; ++r) {
        for (int i = 0; i < count_per_rank; ++i) {
            int t = rank_slots[r][i];
            memcpy(send + ((size_t)r * count_per_rank + i) * hidden,
                   x + (size_t)t * hidden, hidden * sizeof(uint16_t));
        }
    }
    memset(cmb_send, 0, total_bytes);
    free(x);

    g_send = send; g_recv = recv;
    g_cmb_send = cmb_send; g_cmb_recv = cmb_recv;
    g_total_bytes = total_bytes;
    g_mpi_count = (int)(count_per_rank * hidden);
    if ((long)count_per_rank * hidden > INT_MAX) {
        if (rank == 0) fprintf(stderr, "MPI count overflow\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef USE_BFLOAT16_AS_UINT16
    g_elem_type = MPI_UINT16_T;
#else
    g_elem_type = MPI_BFLOAT16_T;
#endif

    /* Sanity dispatch. */
    do_dispatch();
    MPI_Barrier(MPI_COMM_WORLD);

    double *dispatch_us = malloc(iters * sizeof(double));
    double *combine_us = malloc(iters * sizeof(double));

    for (int i = 0; i < warmup; ++i) do_dispatch();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        do_dispatch();
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        dispatch_us[i] = (t1 - t0) * 1e6;
    }

    if (!no_combine) {
        for (int i = 0; i < warmup; ++i) do_combine();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < iters; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            do_combine();
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            combine_us[i] = (t1 - t0) * 1e6;
        }
    } else {
        for (int i = 0; i < iters; ++i) combine_us[i] = 0.0;
    }

    /* Point-to-point pairwise benchmark.
     * For every ordered pair (src, dst) with src != dst, measure a ping-pong
     * of a message whose size equals what one rank sends to one peer in the
     * alltoall.  Pairs are measured sequentially to avoid cross-traffic.
     * All ranks participate in a global barrier before each pair so that the
     * network is quiet for the measurement.
     * The receive is posted with MPI_Irecv before the matching send so that
     * large-message rendezvous handshakes never wait for a receive to appear. */
    int p2p_warmup = 5;
    int p2p_iters = 200;
    size_t p2p_bytes = g_mpi_count * sizeof(uint16_t);
    uint8_t *p2p_send = NULL;
    uint8_t *p2p_recv = NULL;
    double **p2p_us = NULL;
    int *p2p_peer_valid = NULL;

    if (!no_p2p) {
        p2p_send = malloc(p2p_bytes);
        p2p_recv = malloc(p2p_bytes);
        if (!p2p_send || !p2p_recv) {
            fprintf(stderr, "[rank %d] p2p buffer allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(p2p_send, 0xAB, p2p_bytes);

        /* Each rank stores results for pairs where it is the source. */
        p2p_us = calloc(world, sizeof(double *));
        p2p_peer_valid = calloc(world, sizeof(int));
        for (int d = 0; d < world; ++d) {
            if (d == rank) continue;
            p2p_us[d] = malloc(p2p_iters * sizeof(double));
            p2p_peer_valid[d] = 1;
        }

        for (int src = 0; src < world; ++src) {
            for (int dst = 0; dst < world; ++dst) {
                if (src == dst) continue;
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == src) {
                    for (int w = 0; w < p2p_warmup; ++w) {
                        MPI_Request req;
                        MPI_Irecv(p2p_recv, (int)p2p_bytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD, &req);
                        MPI_Send(p2p_send, (int)p2p_bytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                    for (int i = 0; i < p2p_iters; ++i) {
                        MPI_Request req;
                        MPI_Irecv(p2p_recv, (int)p2p_bytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD, &req);
                        double t0 = MPI_Wtime();
                        MPI_Send(p2p_send, (int)p2p_bytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                        double t1 = MPI_Wtime();
                        p2p_us[dst][i] = (t1 - t0) * 1e6;
                    }
                } else if (rank == dst) {
                    for (int w = 0; w < p2p_warmup; ++w) {
                        MPI_Request req;
                        MPI_Irecv(p2p_recv, (int)p2p_bytes, MPI_BYTE, src, 0, MPI_COMM_WORLD, &req);
                        MPI_Send(p2p_send, (int)p2p_bytes, MPI_BYTE, src, 0, MPI_COMM_WORLD);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                    for (int i = 0; i < p2p_iters; ++i) {
                        MPI_Request req;
                        MPI_Irecv(p2p_recv, (int)p2p_bytes, MPI_BYTE, src, 0, MPI_COMM_WORLD, &req);
                        MPI_Send(p2p_send, (int)p2p_bytes, MPI_BYTE, src, 0, MPI_COMM_WORLD);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    }

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hlen;
    MPI_Get_processor_name(hostname, &hlen);

    char cpuinfo_model[256];
    gather_affinity(local_rank, cpuinfo_model, sizeof(cpuinfo_model));

    /* Aggregate hostnames at rank 0. */
    int names_len = (int)strlen(hostname) + 1;
    int *all_name_lens = NULL;
    char *all_names = NULL;
    if (rank == 0) all_name_lens = malloc(world * sizeof(int));
    MPI_Gather(&names_len, 1, MPI_INT, all_name_lens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < world; ++i) total += all_name_lens[i];
        all_names = malloc(total);
        int *displs = malloc(world * sizeof(int));
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
    fprintf(jf, "  \"group_id\": 0,\n");
    fprintf(jf, "  \"world_size\": %d,\n", world);
    fprintf(jf, "  \"ep\": %d,\n", ep);
    fprintf(jf, "  \"chunk_id\": %d,\n", chunk_id);
    fprintf(jf, "  \"total_chunks\": %d,\n", total_chunks);
    fprintf(jf, "  \"pin\": {\"mode\": \"%s\", \"requested\": %s},\n", pin, pin[0] ? "true" : "false");
    fprintf(jf, "  \"affinity\": {\"backend\": \"mpi-cpu\", \"local_rank\": %d, \"cpuinfo_model\": ", local_rank);
    json_string(jf, cpuinfo_model);
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

    /* Ensure every rank has closed and flushed its per-rank JSON before rank 0
     * tries to read them back for the aggregated file. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Aggregate p2p summary across all ranks.  This is a collective operation:
     * every rank must participate in the broadcasts even though only rank 0
     * writes the result to the aggregated JSON file.  If p2p is disabled, all
     * ranks still participate with dummy values so the broadcast tree matches. */
    int pairs = no_p2p ? 0 : world * (world - 1);
    double *all_min = NULL;
    double *all_max = NULL;
    double *all_med = NULL;
    int    *all_src = NULL;
    int    *all_dst = NULL;
    if (pairs > 0) {
        all_min = malloc(pairs * sizeof(double));
        all_max = malloc(pairs * sizeof(double));
        all_med = malloc(pairs * sizeof(double));
        all_src = malloc(pairs * sizeof(int));
        all_dst = malloc(pairs * sizeof(int));
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

    if (p2p_send) free(p2p_send);
    if (p2p_recv) free(p2p_recv);
    if (p2p_us) {
        for (int d = 0; d < world; ++d) free(p2p_us[d]);
        free(p2p_us);
    }
    if (p2p_peer_valid) free(p2p_peer_valid);

    free(send); free(recv); free(cmb_send); free(cmb_recv);
    free(dispatch_us); free(combine_us);
    free(topk_idx); free(send_counts);
    for (int r = 0; r < ep; ++r) free(rank_slots[r]);
    free(rank_slots);
    free(all_name_lens); free(all_names);

    MPI_Finalize();
    return 0;
}
