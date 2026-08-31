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
    lr = (int)getenv_long("SLURM_LOCAL_RANK", -1);
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

static void run_cmd(const char *cmd, char *out, size_t outsz) {
    FILE *p = popen(cmd, "r");
    if (!p) {
        snprintf(out, outsz, "error: %s", strerror(errno));
        return;
    }
    size_t n = fread(out, 1, outsz - 1, p);
    out[n] = '\0';
    int st = pclose(p);
    if (st != 0 && out[0] == '\0') {
        snprintf(out, outsz, "exit %d", st);
    }
    char *end = out + strlen(out) - 1;
    while (end >= out && (*end == '\n' || *end == '\r' || *end == ' ' || *end == '\t')) {
        *end = '\0';
        --end;
    }
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
            double t1 = MPI_Wtime();
            combine_us[i] = (t1 - t0) * 1e6;
        }
    } else {
        for (int i = 0; i < iters; ++i) combine_us[i] = 0.0;
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
    fprintf(jf, "  \"mpi_cpu_staging\": false\n");
    fprintf(jf, "}\n");
    fclose(jf);

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

    free(send); free(recv); free(cmb_send); free(cmb_recv);
    free(dispatch_us); free(combine_us);
    free(topk_idx); free(send_counts);
    for (int r = 0; r < ep; ++r) free(rank_slots[r]);
    free(rank_slots);
    free(all_name_lens); free(all_names);

    MPI_Finalize();
    return 0;
}
