#!/bin/bash
# Sweep the CPU-only MPI-based EP dispatch benchmark across fixed-size node chunks.
# Runs directly in the uenv prgenv-gnu/26.3:v1 with --mpi=cray_shasta.
#
# Usage:
#   bash sweep_ep_dispatch_mpi_cpu_uenv.sh
#   NODES_PER_JOB=16 bash sweep_ep_dispatch_mpi_cpu_uenv.sh
#   RANKS_PER_NODE=4 NODES_PER_JOB=4 bash sweep_ep_dispatch_mpi_cpu_uenv.sh  # EP16, 4 ranks/node
#   OUTTAG=ep-mpi-cpu-uenv-sweep-ep16 bash sweep_ep_dispatch_mpi_cpu_uenv.sh
#   DRY_RUN=true bash sweep_ep_dispatch_mpi_cpu_uenv.sh
#   USE_AVAILABLE=true bash sweep_ep_dispatch_mpi_cpu_uenv.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$HERE/submit_ep_dispatch_mpi_cpu_uenv.sh"

RESV="${RESV:-SD-69241-apertus-1-5-0}"
DRY_RUN="${DRY_RUN:-false}"
START="${START:-0}"
COUNT="${COUNT:-0}"
NODES="${NODES:-}"
NODES_PER_JOB="${NODES_PER_JOB:-4}"
RANKS_PER_NODE="${RANKS_PER_NODE:-1}"
PARTITION="${PARTITION:-normal}"
STRIDE="${STRIDE:-0}"
SEED="${SEED:-}"
USE_AVAILABLE="${USE_AVAILABLE:-true}"
OUTTAG="${OUTTAG:-ep-mpi-cpu-uenv-sweep}/"
NUMA_PIN_MODE="${NUMA_PIN_MODE:-none}"
MPI_NET="${MPI_NET:-cxi}"
GATHER_CXI="${GATHER_CXI:-0}"
GATHER_CXI_COUNTERS_LEVEL="${GATHER_CXI_COUNTERS_LEVEL:-5}"

[ "$NODES_PER_JOB" -ge 1 ] || { echo "NODES_PER_JOB must be >= 1" >&2; exit 1; }
[ "$RANKS_PER_NODE" -ge 1 ] || { echo "RANKS_PER_NODE must be >= 1" >&2; exit 1; }
[ "$STRIDE" -eq 0 ] && STRIDE=$NODES_PER_JOB
[ "$STRIDE" -ge 1 ] || { echo "STRIDE must be >= 1" >&2; exit 1; }

EP=$((NODES_PER_JOB * RANKS_PER_NODE))

if [ -n "$NODES" ]; then
    if [ -f "$NODES" ]; then
        hostlist=$(sed 's/#.*//' "$NODES" | tr -s '[:space:]' '\n' | grep . | paste -sd,)
    else
        hostlist=$NODES
    fi
    src="NODES=$NODES"
else
    hostlist=$(scontrol show reservation "$RESV" | tr ' ' '\n' | grep -m1 '^Nodes=' | cut -d= -f2)
    src="reservation $RESV"
fi
[ -n "$hostlist" ] || { echo "$src: no nodes found" >&2; exit 1; }

if [ "$USE_AVAILABLE" = true ]; then
    mapfile -t available_nodes < <(sinfo -h -N -n "$hostlist" -p "$PARTITION" -o "%N %t" \
        | awk '{gsub(/[*$~#!%]+$/, "", $2); if ($2 ~ /^(idle|resv)$/) print $1}' \
        | sort -u)
    if [ ${#available_nodes[@]} -lt "$NODES_PER_JOB" ]; then
        echo "USE_AVAILABLE=true: only ${#available_nodes[@]} idle/reserved nodes, need $NODES_PER_JOB" >&2
        exit 1
    fi
    hostlist=$(IFS=,; echo "${available_nodes[*]}")
    src="$src (idle/reserved only)"
fi

mapfile -t nodes < <(sinfo -h -N -n "$hostlist" -p "$PARTITION" -o "%N %t" \
    | awk '{gsub(/[*$~#!%]+$/, "", $2); if ($2 ~ /^(idle|alloc|mix|resv|comp|plnd)$/) print $1}' \
    | sort -u)
total=${#nodes[@]}
[ "$total" -ge "$NODES_PER_JOB" ] || {
    echo "only $total usable nodes in $src, need $NODES_PER_JOB" >&2; exit 1;
}

if [ -n "$SEED" ]; then
    mapfile -t nodes < <(printf '%s\n' "${nodes[@]}" | shuf --random-source=<(yes "$SEED"))
fi

nodes+=("${nodes[@]:0:$((NODES_PER_JOB - 1))}")
chunks=$(((total + STRIDE - 1) / STRIDE))

last=$chunks
if [ "$COUNT" -gt 0 ] && [ "$((START + COUNT))" -lt "$chunks" ]; then
    last=$((START + COUNT))
fi

echo "$src: $total usable nodes in $PARTITION -> $chunks groups of $NODES_PER_JOB nodes x $RANKS_PER_NODE ranks (EP$EP), stride $STRIDE, submitting [$START,$last)"
echo "  transport: $MPI_NET  NUMA pin: $NUMA_PIN_MODE  gather CXI: $GATHER_CXI"

OUTDIR="$HERE/slurm_logs"
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/ep-mpi-cpu-uenv-manifest-$(date +%Y%m%d-%H%M%S).txt"

for ((i = START; i < last; i++)); do
    chunk=$(IFS=,; echo "${nodes[*]:i*STRIDE:NODES_PER_JOB}")

    if [ "$DRY_RUN" = true ]; then
        printf 'chunk %3d %s\n' "$i" "$chunk"
        continue
    fi

    jid=$(CHUNK_ID="$i" TOTAL_CHUNKS="$chunks" EP="$EP" OUTTAG="$OUTTAG" \
          RANKS_PER_NODE="$RANKS_PER_NODE" \
          NUMA_PIN_MODE="$NUMA_PIN_MODE" MPI_NET="$MPI_NET" \
          GATHER_CXI="$GATHER_CXI" GATHER_CXI_COUNTERS_LEVEL="$GATHER_CXI_COUNTERS_LEVEL" \
          sbatch --parsable \
          --nodes="$NODES_PER_JOB" \
          --ntasks-per-node="$RANKS_PER_NODE" \
          --cpus-per-task=72 \
          --partition="$PARTITION" --nodelist="$chunk" \
          "$SBATCH_FILE")
    echo "$jid $chunk" >>"$MANIFEST"
    printf 'chunk %3d job %s  %s\n' "$i" "$jid" "$chunk"
done

if [ "$DRY_RUN" != true ]; then
    echo
    echo "manifest $MANIFEST"
    echo "watch    squeue -u $USER -n ep-dispatch-mpi-cpu-uenv"
    echo "abort    scancel -u $USER -n ep-dispatch-mpi-cpu-uenv"
    echo "report   python3 $HERE/report_ep_dispatch_custom.py --outdir '${OUTTAG}results-ep-mpi-cpu-uenv-*'"
    echo "note     each result dir contains $RANKS_PER_NODE rank JSONs per node"
fi
