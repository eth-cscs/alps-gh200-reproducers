# MoE EP dispatch/combine all-to-all latency reproducer

This directory contains a minimal CPU-only reproducer for intermittent high
latency observed in the Expert Parallelism (EP) dispatch/combine
`all_to_all_single` operations on CSCS Alps GH200 nodes.  The goal is to
determine whether the variability is a property of the node pair, the Slingshot
CXI fabric, the container/MPI environment, or external system monitoring.

## Problem statement

A workload structured as an MoE EP group performs an MPI `Alltoall`
(representing the dispatch and combine phases of `all_to_all_single`).  On the
same pair of nodes, repeated runs show a bimodal latency distribution: many
iterations complete in ~10–11 ms, while others take ~20–60 ms.  The slow mode
appears to be a transient allocation/initialization effect rather than a stable
node property: some repeats are uniformly fast, others are uniformly slow, and
within a slow repeat both ranks are slow together.

## Files

| File | Purpose |
|------|---------|
| `ep_dispatch_bench_mpi_cpu.c` | CPU-only EP dispatch/combine benchmark using MPI `Alltoall`. |
| `ep_dispatch_bench_mpi_cpu_cnt` | Pre-built binary (built inside the container on first run if source is newer). |
| `Makefile.ep_mpi_cpu_cnt` | Local Makefile for manual builds with `mpicc`. |
| `submit_ep_dispatch_mpi_cpu_container.sh` | Slurm submit script that runs the benchmark inside an Apptainer/Sarus container via `--environment=<toml>` and `--mpi=pmix`. |
| `sweep_ep_dispatch_mpi_cpu_pair_repeat.sh` | Repeatedly submits the benchmark on the same two nodes to test run-to-run variability. |
| `report_ep_dispatch_custom.py` | Aggregates per-rank JSON outputs and flags slow chunks/nodes. |
| `alps-pytorch2602.toml` | Container environment for Alps6 (single-node/multi-rank tests). |
| `alps4-pytorch2602.toml` | Container environment for Alps4 (multi-node tests). |
| `pair-repeat-slow-r13/` | Example output from 20 repeats on a known slow node pair. |
| `slurm_logs/` | Slurm stdout/stderr and optional per-rank libfabric logs. |

## Quick start

### 1. Single 4-node run

```bash
sbatch submit_ep_dispatch_mpi_cpu_container.sh
```

The default sbatch header uses 4 nodes, 1 rank per node, 72 CPUs per rank.
Results land in `results-ep-mpi-cpu-ctn-<jobid>/`.

### 2. Repeated pair test

Pick a node pair known to show variability and run 20 independent jobs:

```bash
NODES=nid006065,nid006066 N_REPEATS=20 bash sweep_ep_dispatch_mpi_cpu_pair_repeat.sh
```

This writes per-job directories under `ep-mpi-cpu-ctn-pair-repeat/` and a
manifest in `slurm_logs/`.

### 3. Analyze results

```bash
python3 report_ep_dispatch_custom.py --outdir 'ep-mpi-cpu-ctn-pair-repeat/results-ep-mpi-cpu-ctn-*'
```

This prints per-chunk and per-node median dispatch latencies and flags chunks
>1.5× the overall median.

## Benchmark details

`ep_dispatch_bench_mpi_cpu.c` is a stripped-down CPU-only version of the EP
dispatch/combine pattern:

- Default shape: 8192 tokens, hidden size 1792, top-k 8, 256 experts, EP 8.
- Each rank builds a balanced top-k routing table and permutes token slices into
  an `MPI_Alltoall` send buffer.
- Two phases are timed: **dispatch** (send tokens to experts) and **combine**
  (scatter/gather reverse).  Use `--no-combine` to skip the combine phase.
- Per-iteration timings are written as JSON arrays (`dispatch_us`,
  `combine_us`).

Command-line options (passed through by the submit script):

```text
--num-tokens N     number of tokens (default: 8192)
--hidden H         hidden size (default: 1792)
--num-topk K       top-k (default: 8)
--num-experts E    total experts (default: 256)
--ep E             EP degree; must divide world size (default: 8)
--iters N          timed iterations (default: 200)
--warmup N         warmup iterations (default: 10)
--seed N           RNG seed (default: 0)
--chunk-id N       chunk id for sweep bookkeeping (default: 0)
--total-chunks N   total chunks in sweep (default: 1)
--no-combine       skip combine phase
--outdir DIR       output directory (default: results-ep-mpi-cpu)
--out FILE         aggregated JSON output at rank 0
--pin SPEC         CPU pinning spec (passed through to benchmark)
--balanced-routing ignored; routing is always balanced
```

## Environment variables

The submit script and sweep script honor the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_FILE` | `./alps-pytorch2602.toml` | Container TOML to use. |
| `TIMES` | 200 | Benchmark iterations. |
| `WARMUP` | 10 | Warmup iterations. |
| `OMP_NUM_THREADS` | 72 | OpenMP threads. |
| `PIN_RANKS` | "" | CPU pinning spec passed to benchmark. |
| `CXI_DEV_PER_RANK` | "" | Comma-separated `cxi0,cxi1,...` device names to pin per local rank. |
| `GATHER_CXI` | 0 | Wrap ranks with `gather_cxi_counters` (requires the binary in `$PWD`). |
| `GATHER_CXI_COUNTERS_LEVEL` | 5 | Counter detail level. |
| `GATHER_CXI_DETAILED` | 0 | Enable time-series sampling inside the wrapper. |
| `GATHER_CXI_INTERVAL` | 100 | Sample interval when `DETAILED=1` (ms). |
| `LF_LOG_LEVEL` | "" | Set to `debug` to capture libfabric logging per rank. |
| `LF_LOG_PROV` | `cxi` | Provider filter for libfabric logging. |
| `OUTTAG` | "" | Output directory prefix. |

## Known issues and caveats

1. **Hardcoded `FI_CXI_RX_MATCH_MODE=software` was removed.**  Earlier versions
   of the submit script forced software RX matching, which inflated latency
   uniformly.  The current script leaves the default (`hybrid`) unless you set
   it explicitly.

2. **`GATHER_CXI_OS_METRICS` is disabled by default.**  The wrapper only reads
   CXI telemetry counters, not inherited `perf` counters.  Enabling OS metrics
   (`GATHER_CXI_OS_METRICS=1`) adds inherited perf counter overhead and has been
   observed to shift the whole latency distribution upward.

3. **Wrap only one rank per node if possible.**  The current wrapper wraps every
   rank when `GATHER_CXI=1`.  Because CXI counters are per-NIC, one rank per
   node is sufficient and reduces file-descriptor / process overhead.

4. **External monitoring can interfere.**  LDMS, `gather_cxi_counters`,
   `perf_event_open`-based tools, or any process continuously reading
   `/sys/class/net/hsn*/device/telemetry/` can contend with MPI/CXI progress and
   raise median latency.  The samplers in the observed LDMS configuration run at
   10-second intervals, which is too slow to explain the per-iteration uniform
   shift, but they can still cause occasional outliers.

5. **Container vs uenv.**  This reproducer intentionally mirrors the uenv MPI
   CPU workflow but uses a container via `--environment=<toml>` so the same
   binary can be tested with the same PyTorch/MPI stack as the original
   workload.

## Typical workflow for debugging slow nodes

1. Identify a slow node pair from production logs or from a previous sweep.
2. Run the pair-repeat sweep on that pair with `GATHER_CXI=0`:
   ```bash
   NODES=nid006065,nid006066 N_REPEATS=20 GATHER_CXI=0 \
       bash sweep_ep_dispatch_mpi_cpu_pair_repeat.sh
   ```
3. Analyze with `report_ep_dispatch_custom.py`.  If repeats are still bimodal,
   the cause is likely the node pair, fabric routing, or external monitoring.
4. Re-run with `GATHER_CXI=1` to collect CXI counters on slow vs fast repeats.
5. Compare `cxi-counters-*-rank*.json` files and Slurm logs between slow and
   fast repeats.

## Capturing libfabric debug logs

Set `LF_LOG_LEVEL=debug` when submitting:

```bash
LF_LOG_LEVEL=debug LF_LOG_PROV=cxi \
    sbatch submit_ep_dispatch_mpi_cpu_container.sh
```

Each rank's stderr is redirected to
`slurm_logs/ep-dispatch-mpi-cpu-ctn-<jobid>-logs/rank<N>.log`.  These can be
compared across slow and fast runs to spot provider decisions that differ.

## Contact / context

This reproducer was created for CSCS ticket SD-70419 to isolate EP dispatch
latency variability on Alps Aptus GH200 nodes.  The broader investigation
includes CXI telemetry, LDMS monitoring, node diagnostics, and comparisons
between uenv and container execution.
