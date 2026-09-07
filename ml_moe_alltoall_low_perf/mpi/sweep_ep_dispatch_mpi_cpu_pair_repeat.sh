#!/bin/bash
# Repeatedly submit the container-based CPU-only MPI EP dispatch benchmark on
# the same pair of nodes. This tests whether slowness is a stable node property
# or depends on allocation/initialization state.
#
# Usage:
#   NODES=nid007231,nid007233 N_REPEATS=20 bash sweep_ep_dispatch_mpi_cpu_pair_repeat.sh
#
# Optional environment variables:
#   NODES           comma-separated pair of nodes (required)
#   N_REPEATS       number of independent jobs to submit (default: 20)
#   RANKS_PER_NODE  ranks per node (default: 1)
#   EP              world size; default = 2 * RANKS_PER_NODE
#   TIMES           iterations per job (default: 50)
#   WARMUP          warmup iterations (default: 5)
#   ENV_FILE        container TOML (default: ./alps-pytorch2602.toml)
#   OUTTAG          output prefix passed to the submit script (default:
#                   ep-mpi-cpu-ctn-pair-repeat/); result dirs will be
#                   ${OUTTAG}results-ep-mpi-cpu-ctn-<jobid>
#   DRY_RUN         if "true", only print what would be submitted
#   SHUFFLE_RANKS   if "true", alternate which node is rank 0/1 across repeats
#   GATHER_CXI              set to 1 to wrap each rank with gather_cxi_counters
#                           (default: 0)
#   GATHER_CXI_COUNTERS_LEVEL  counter detail level (default: 5)
#
# Output:
#   Per-job result directories under ${OUTTAG}/.
#   A manifest in slurm_logs/ listing job IDs and node order.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$HERE/submit_ep_dispatch_mpi_cpu_container.sh"

NODES="${NODES:-}"
N_REPEATS="${N_REPEATS:-20}"
RANKS_PER_NODE="${RANKS_PER_NODE:-1}"
EP="${EP:-$((2 * RANKS_PER_NODE))}"
TIMES="${TIMES:-50}"
WARMUP="${WARMUP:-5}"
ENV_FILE="${ENV_FILE:-./alps-pytorch2602.toml}"
OUTTAG="${OUTTAG:-ep-mpi-cpu-ctn-pair-repeat}"
DRY_RUN="${DRY_RUN:-false}"
SHUFFLE_RANKS="${SHUFFLE_RANKS:-false}"
GATHER_CXI="${GATHER_CXI:-0}"
GATHER_CXI_COUNTERS_LEVEL="${GATHER_CXI_COUNTERS_LEVEL:-5}"

[ -n "$NODES" ] || { echo "NODES is required (comma-separated pair)" >&2; exit 1; }

# Normalize to exactly two nodes.
IFS=',' read -ra NODE_ARR <<< "$NODES"
[ ${#NODE_ARR[@]} -eq 2 ] || { echo "NODES must contain exactly two nodes, got ${#NODE_ARR[@]}" >&2; exit 1; }

# Normalize OUTTAG so it always behaves like the other sweep scripts: results
# land in ${OUTTAG}results-ep-mpi-cpu-ctn-<jobid>. If the user gives a bare
# prefix without a trailing slash, append one for a clean subdirectory.
if [ "${OUTTAG: -1}" != "/" ]; then
    OUTTAG="${OUTTAG}/"
fi
mkdir -p "$HERE/slurm_logs" "$OUTTAG"
MANIFEST="$HERE/slurm_logs/ep-mpi-cpu-ctn-pair-repeat-$(date +%Y%m%d-%H%M%S).txt"

echo "Submitting $N_REPEATS independent jobs on ${NODE_ARR[0]} + ${NODE_ARR[1]}"
echo "  ranks per node: $RANKS_PER_NODE, EP: $EP, iters: $TIMES, warmup: $WARMUP"
echo "  container: $ENV_FILE"
echo "  shuffle ranks: $SHUFFLE_RANKS"
echo "  gather CXI counters: $GATHER_CXI"
[ "$GATHER_CXI" != "0" ] && echo "  GATHER_CXI_COUNTERS_LEVEL: $GATHER_CXI_COUNTERS_LEVEL"
echo "  manifest: $MANIFEST"

for ((i = 0; i < N_REPEATS; i++)); do
    if [ "$SHUFFLE_RANKS" = true ] && [ $((i % 2)) -eq 1 ]; then
        chunk="${NODE_ARR[1]},${NODE_ARR[0]}"
    else
        chunk="${NODE_ARR[0]},${NODE_ARR[1]}"
    fi

    if [ "$DRY_RUN" = true ]; then
        printf 'repeat %3d %s\n' "$i" "$chunk"
        continue
    fi

    jid=$(CHUNK_ID="$i" TOTAL_CHUNKS="$N_REPEATS" EP="$EP" \
          OUTTAG="$OUTTAG" \
          RANKS_PER_NODE="$RANKS_PER_NODE" \
          ENV_FILE="$ENV_FILE" TIMES="$TIMES" WARMUP="$WARMUP" \
          GATHER_CXI="$GATHER_CXI" \
          GATHER_CXI_COUNTERS_LEVEL="$GATHER_CXI_COUNTERS_LEVEL" \
          sbatch --parsable \
          --nodes=2 \
          --ntasks-per-node="$RANKS_PER_NODE" \
          --cpus-per-task=72 \
          --partition=normal \
          --nodelist="$chunk" \
          "$SBATCH_FILE")
    echo "$jid $i $chunk" >>"$MANIFEST"
    printf 'repeat %3d job %s  %s\n' "$i" "$jid" "$chunk"
done

if [ "$DRY_RUN" != true ]; then
    echo
    echo "manifest $MANIFEST"
    echo "watch    squeue -u $USER -n ep-dispatch-mpi-cpu-ctn"
    echo "abort    scancel -u $USER -n ep-dispatch-mpi-cpu-ctn"
    echo "report   python3 $HERE/report_ep_dispatch_custom.py --outdir '${OUTTAG}results-ep-mpi-cpu-ctn-*'"
fi
