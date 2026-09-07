#!/bin/bash
#SBATCH --job-name=ep-dispatch-mpi-cpu-ctn
#SBATCH --account=csstaff
# BATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --reservation=SD-70757-apertus-node-diag
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/ep-dispatch-mpi-cpu-ctn-%j.out
#SBATCH --error=slurm_logs/ep-dispatch-mpi-cpu-ctn-%j.err

set -euo pipefail

# CPU-only MPI EP dispatch benchmark (C binary, supports multiple ranks per node).
# Runs inside an Apptainer/Sarus container (via --environment) instead of a uenv.
# MPI bootstrap uses pmix as in the NCCL/PyTorch container workflow.
#
# Usage:
#   sbatch submit_ep_dispatch_mpi_cpu_container.sh
#   sbatch --nodelist=nid006106,nid006108,... submit_ep_dispatch_mpi_cpu_container.sh
#   sbatch --nodes=1 --ntasks-per-node=4 submit_ep_dispatch_mpi_cpu_container.sh  # single-node EP4
#   sbatch --nodes=2 --ntasks-per-node=2 submit_ep_dispatch_mpi_cpu_container.sh  # 2-node EP4

export OUTTAG=${OUTTAG:-}
export TIMES=${TIMES:-200}
export WARMUP=${WARMUP:-10}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export OUTDIR="${OUTDIR:-${OUTTAG}results-ep-mpi-cpu-ctn-${SLURM_JOB_ID:-0}}"
export PIN_RANKS=${PIN_RANKS:-""}

export CHUNK_ID="${CHUNK_ID:-0}"
export TOTAL_CHUNKS="${TOTAL_CHUNKS:-1}"

# Container environment TOML. Override via ENV_FILE when submitting/sweeping.
export ENV_FILE="${ENV_FILE:-./alps-pytorch2602.toml}"

# Optional libfabric debug logging. Set LF_LOG_LEVEL=debug and optionally
# LF_LOG_PROV=cxi to capture provider decisions per rank.
export LF_LOG_LEVEL="${LF_LOG_LEVEL:-}"
export LF_LOG_PROV="${LF_LOG_PROV:-}"

# Optional per-rank CXI device pinning. Comma-separated list mapped by
# LOCAL_RANK. Example: CXI_DEV_PER_RANK="cxi0,cxi1,cxi2,cxi3" pins local rank
# 0 to cxi0, rank 1 to cxi1, etc. Useful to force same-switch vs cross-switch
# routing on the same node pair.
export CXI_DEV_PER_RANK="${CXI_DEV_PER_RANK:-}"

# Optional NUMA pinning mode for each rank.
#   none  - no numactl pinning (default)
#   last  - pin rank to the last CPU of its assigned NUMA node
#   node  - bind rank to all CPUs and memory of its assigned NUMA node
# The assigned node is NUMA_NODE_BASE + LOCAL_RANK.  On GH200 with 72 CPUs per
# NUMA node, NUMA_PIN_MODE=last pins local rank 0 to CPU 71 of node 0, rank 1
# to CPU 143 of node 1, etc.
export NUMA_PIN_MODE="${NUMA_PIN_MODE:-none}"
export NUMA_NODE_BASE="${NUMA_NODE_BASE:-0}"
export NUMA_CPUS_PER_NODE="${NUMA_CPUS_PER_NODE:-72}"

# Optional network transport selection for Cray MPICH / libfabric.
#   cxi     - use Slingshot CXI provider (default)
#   tcp     - force libfabric TCP provider
#   sockets - force libfabric sockets provider
# This is useful as a negative control to verify whether an issue is specific
# to the CXI/Slingshot path.  When tcp/sockets is selected, CXI is excluded
# explicitly so libfabric cannot fall back to it.
export MPI_NET="${MPI_NET:-cxi}"

# Optional CXI counter collection. Set GATHER_CXI=1 to wrap each rank with
# gather_cxi_counters. Output lands next to the benchmark JSON.
export GATHER_CXI="${GATHER_CXI:-0}"
export GATHER_CXI_COUNTERS_BIN="${GATHER_CXI_COUNTERS_BIN:-$PWD/gather_cxi_counters}"
# gather_cxi_counters collects all counters by default; level controls sample
# interval/detail (default 5).
export GATHER_CXI_COUNTERS_LEVEL="${GATHER_CXI_COUNTERS_LEVEL:-5}"
export GATHER_CXI_DETAILED="${GATHER_CXI_DETAILED:-0}"
export GATHER_CXI_INTERVAL="${GATHER_CXI_INTERVAL:-100}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS

mkdir -p "$OUTDIR" slurm_logs

echo "[$(date --iso-8601=seconds)] job $SLURM_JOB_ID  $SLURM_NNODES nodes  $WORLD_SIZE ranks  chunk $CHUNK_ID/$TOTAL_CHUNKS"
echo "nodes: $(scontrol show hostlistsorted "$SLURM_JOB_NODELIST")"
export BENCH_VARIANT="${BENCH_VARIANT:-dispatch}"
if [ "$BENCH_VARIANT" = "p2p" ]; then
    export BENCH_SOURCE="$PWD/ep_dispatch_bench_mpi_cpu_p2p.c"
    export BENCH_BINARY="$PWD/ep_dispatch_bench_mpi_cpu_p2p_cnt"
else
    export BENCH_SOURCE="$PWD/ep_dispatch_bench_mpi_cpu.c"
    export BENCH_BINARY="$PWD/ep_dispatch_bench_mpi_cpu_cnt"
fi
echo "binary: $BENCH_BINARY"
echo "environment: $ENV_FILE"

# Build the binary inside the container if it is missing or the source is newer.
# This runs on a single task inside the allocation so the container toolchain
# (mpicc, headers, MPI libs) is used.
if [ ! -x "$BENCH_BINARY" ] || [ "$BENCH_SOURCE" -nt "$BENCH_BINARY" ]; then
    echo "building $BENCH_SOURCE inside container..."
    srun --environment="$ENV_FILE" \
         --mpi=pmix \
         --network=disable_rdzv_get \
         --ntasks=1 --nodes=1 --cpus-per-task=1 \
         bash -c "mpicc -O2 -std=gnu11 -Wall -o '$BENCH_BINARY' '$BENCH_SOURCE'"
    echo "build done"
fi

LAUNCHER="$PWD/.ep_dispatch_mpi_cpu_ctn_launcher.$SLURM_JOB_ID.sh"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
set -euo pipefail
export LOCAL_RANK=\${SLURM_LOCALID:-0}
export RANK=\$SLURM_PROCID
export WORLD_SIZE=\$SLURM_NPROCS

#export FI_CXI_RX_MATCH_MODE=software

# Optional libfabric debug logging. When enabled, redirect each rank's stderr
# to a dedicated log file under slurm_logs/<job>/ so logs for one job are kept
# together and easy to diff as a group.
if [ -n "\${LF_LOG_LEVEL:-}" ]; then
    export FI_LOG_LEVEL="\$LF_LOG_LEVEL"
    export FI_LOG_PROV="\${LF_LOG_PROV:-cxi}"
    LOGDIR="$PWD/slurm_logs/ep-dispatch-mpi-cpu-ctn-\$SLURM_JOB_ID-logs"
    mkdir -p "\$LOGDIR"
    exec 2>"\$LOGDIR/rank\$RANK.log"
fi

BENCH="$BENCH_BINARY"
BENCH_ARGS=(
    --ep "\$SLURM_NPROCS"
    --iters "$TIMES"
    --warmup "$WARMUP"
    --chunk-id "$CHUNK_ID"
    --total-chunks "$TOTAL_CHUNKS"
    --balanced-routing
    --pin "$PIN_RANKS"
    --outdir "$OUTDIR"
    --out "$OUTDIR/ep-dispatch-\$SLURM_JOB_ID.json"
    "\$@"
)

# Build optional NUMA pinning prefix.
NUMA_PREFIX=()
# Apply Cray MPICH / libfabric transport selection.
case "\${MPI_NET:-cxi}" in
    tcp)
        unset FI_PROVIDER
        export MPICH_OFI_USE_PROVIDER="tcp;ofi_rxm"
        export MPICH_OFI_ENABLE_HMEM=0
        export MPICH_OFI_STARTUP_CONNECT=1
        export MPICH_OFI_VERBOSE=1
        ;;
    sockets)
        unset FI_PROVIDER
        export MPICH_OFI_USE_PROVIDER=sockets
        export MPICH_OFI_ENABLE_HMEM=0
        export MPICH_OFI_VERBOSE=1
        ;;
    cxi|*)
        # Default Slingshot/CXI path.  Explicit CXI device selection is applied
        # below if CXI_DEV_PER_RANK is set.
        ;;
esac

# Apply per-rank CXI device selection if requested *and* we are on the CXI path.
if [ "\${MPI_NET:-cxi}" = "cxi" ] && [ -n "\${CXI_DEV_PER_RANK:-}" ]; then
    IFS=',' read -ra CXI_DEVS <<< "\$CXI_DEV_PER_RANK"
    if [ "\$LOCAL_RANK" -lt "\${#CXI_DEVS[@]}" ]; then
        export FI_CXI_DEVICE_NAME="\${CXI_DEVS[\$LOCAL_RANK]}"
    else
        echo "warning: LOCAL_RANK=\$LOCAL_RANK out of range for CXI_DEV_PER_RANK" >&2
    fi
fi

case "\${NUMA_PIN_MODE:-none}" in
    last)
        _NODE=\$((NUMA_NODE_BASE + \${SLURM_LOCALID:-0}))
        _LAST_CPU=\$(( (_NODE + 1) * NUMA_CPUS_PER_NODE - 1 ))
        NUMA_PREFIX=(numactl --physcpubind="\$_LAST_CPU" --membind="\$_NODE")
        ;;
    node)
        _NODE=\$((NUMA_NODE_BASE + \${SLURM_LOCALID:-0}))
        NUMA_PREFIX=(numactl --cpunodebind="\$_NODE" --membind="\$_NODE")
        ;;
    none|*)
        NUMA_PREFIX=()
        ;;
esac

if [ -n "\${GATHER_CXI:-}" ] && [ "\${GATHER_CXI}" != "0" ]; then
    export GATHER_CXI_COUNTERS_LEVEL="\${GATHER_CXI_COUNTERS_LEVEL:-5}"
    export GATHER_CXI_DETAILED="\${GATHER_CXI_DETAILED:-0}"
    export GATHER_CXI_INTERVAL="\${GATHER_CXI_INTERVAL:-100}"
    export GATHER_CXI_JSON=1
    export ZMQ_PORT=\$((12345 + \${SLURM_LOCALID:-0}))
    export GATHER_CXI_OS_METRICS=0
    export GATHER_CXI_INTERVAL=100000
    CTR_TAG="pair-\$SLURM_JOB_ID-r\$RANK"
    CTR_OUT="$OUTDIR/cxi-counters-\$SLURM_JOB_ID-rank\$RANK.json"
    # gather_cxi_counters only emits counter output on local_rank 0. Other ranks
    # produce empty stdout, which is fine; we still keep the file placeholder.
    if [ "\${SLURM_LOCALID:-0}" -eq 0 ]; then
        "\${NUMA_PREFIX[@]}" "$GATHER_CXI_COUNTERS_BIN" -e "\$CTR_TAG" "\$BENCH" "\${BENCH_ARGS[@]}" > "\$CTR_OUT" 2> "\$CTR_OUT.err"
    else
        "\${NUMA_PREFIX[@]}" "$GATHER_CXI_COUNTERS_BIN" -e "\$CTR_TAG" "\$BENCH" "\${BENCH_ARGS[@]}" > /dev/null 2> "\$CTR_OUT.err"
    fi
else
    CTR_OUT="$OUTDIR/bench-\$SLURM_JOB_ID-rank\$RANK"
    "\${NUMA_PREFIX[@]}" "\$BENCH" "\${BENCH_ARGS[@]}" > "\${CTR_OUT}.out" 2> "\${CTR_OUT}.err"
fi
EOF
chmod +x "$LAUNCHER"

cleanup() { rm -f "$LAUNCHER"; }
trap cleanup EXIT

# Slurm will fill SLURM_NTASKS_PER_NODE from the sbatch directive.
export TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
export CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-72}"

srun --environment="$ENV_FILE" \
     --mpi=pmix \
     --network=disable_rdzv_get \
     --ntasks-per-node="$TASKS_PER_NODE" \
     --cpus-per-task="$CPUS_PER_TASK" \
     --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
     "$LAUNCHER" "$@"
