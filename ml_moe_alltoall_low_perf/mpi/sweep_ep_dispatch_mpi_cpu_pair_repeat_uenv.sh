#!/bin/bash
# Repeatedly submit the uenv CPU-only MPI EP dispatch benchmark on the same
# node(s).  This tests whether slowness is a stable node property or depends on
# allocation/initialization state.
#
# Runs directly in the uenv prgenv-gnu/26.3:v1 with --mpi=cray_shasta.
#
# Usage:
#   NODES=nid007231,nid007233 N_REPEATS=20 bash sweep_ep_dispatch_mpi_cpu_pair_repeat_uenv.sh
#   NODES=nid007231 N_REPEATS=20 RANKS_PER_NODE=4 bash sweep_ep_dispatch_mpi_cpu_pair_repeat_uenv.sh
#
# Optional environment variables:
#   NODES           comma-separated node list (1 or 2 nodes; required)
#   N_REPEATS       number of independent jobs to submit (default: 20)
#   RANKS_PER_NODE  ranks per node (default: 1 for 2-node, EP for 1-node)
#   EP              world size; default = nodes * RANKS_PER_NODE
#   TIMES           iterations per job (default: 50)
#   WARMUP          warmup iterations (default: 5)
#   OUTTAG          output prefix passed to the submit script (default:
#                   ep-mpi-cpu-uenv-pair-repeat/)
#   BENCH_VARIANT   dispatch | p2p | nccl  (default: dispatch)
#                   nccl uses ep_dispatch_bench_nccl_p2p.cu and runs on GPU.
#   DRY_RUN         if "true", only print what would be submitted
#   SHUFFLE_RANKS   if "true", alternate which node is rank 0/1 across repeats
#                   (only meaningful with two nodes)
#   NUMA_PIN_MODE   none | last | node  (default: none)
#   MPI_NET         cxi | tcp | sockets (default: cxi)
#   GATHER_CXI              set to 1 to wrap each rank with gather_cxi_counters
#                           (default: 0)
#   GATHER_CXI_COUNTERS_LEVEL  counter detail level (default: 5)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$HERE/submit_ep_dispatch_mpi_cpu_uenv.sh"

NODES="${NODES:-}"
N_REPEATS="${N_REPEATS:-20}"
RANKS_PER_NODE="${RANKS_PER_NODE:-}"
TIMES="${TIMES:-200}"
WARMUP="${WARMUP:-5}"
OUTTAG="${OUTTAG:-ep-mpi-cpu-uenv-pair-repeat}"
BENCH_VARIANT="${BENCH_VARIANT:-dispatch}"
DRY_RUN="${DRY_RUN:-false}"
SHUFFLE_RANKS="${SHUFFLE_RANKS:-false}"
NUMA_PIN_MODE="${NUMA_PIN_MODE:-none}"
MPI_NET="${MPI_NET:-cxi}"
GATHER_CXI="${GATHER_CXI:-0}"
GATHER_CXI_COUNTERS_LEVEL="${GATHER_CXI_COUNTERS_LEVEL:-5}"

[ -n "$NODES" ] || { echo "NODES is required (comma-separated list of 1 or 2 nodes)" >&2; exit 1; }

# Normalize to 1 or 2 nodes.
IFS=',' read -ra NODE_ARR <<< "$NODES"
N_NODE="${#NODE_ARR[@]}"
[ "$N_NODE" -eq 1 ] || [ "$N_NODE" -eq 2 ] || { echo "NODES must contain 1 or 2 nodes, got $N_NODE" >&2; exit 1; }

# Default ranks per node depends on single vs pair mode.
if [ -z "$RANKS_PER_NODE" ]; then
    if [ "$N_NODE" -eq 1 ]; then
        RANKS_PER_NODE=4
    else
        RANKS_PER_NODE=1
    fi
fi
EP="${EP:-$((N_NODE * RANKS_PER_NODE))}"

# Normalize OUTTAG so it always behaves like the other sweep scripts: results
# land in ${OUTTAG}/results-ep-mpi-cpu-uenv-<jobid>.
if [ "${OUTTAG: -1}" != "/" ]; then
    OUTTAG="${OUTTAG}/"
fi
mkdir -p "$HERE/slurm_logs" "$OUTTAG"
MANIFEST="$HERE/slurm_logs/ep-mpi-cpu-uenv-pair-repeat-$(date +%Y%m%d-%H%M%S).txt"

if [ "$N_NODE" -eq 1 ]; then
    echo "Submitting $N_REPEATS independent jobs on single node ${NODE_ARR[0]}"
else
    echo "Submitting $N_REPEATS independent jobs on ${NODE_ARR[0]} + ${NODE_ARR[1]}"
fi
echo "  ranks per node: $RANKS_PER_NODE, EP: $EP, iters: $TIMES, warmup: $WARMUP"
echo "  variant: $BENCH_VARIANT  transport: $MPI_NET  NUMA pin: $NUMA_PIN_MODE  shuffle ranks: $SHUFFLE_RANKS"
echo "  gather CXI counters: $GATHER_CXI"
[ "$GATHER_CXI" != "0" ] && echo "  GATHER_CXI_COUNTERS_LEVEL: $GATHER_CXI_COUNTERS_LEVEL"
echo "  manifest: $MANIFEST"

for ((i = 0; i < N_REPEATS; i++)); do
    if [ "$N_NODE" -eq 1 ]; then
        chunk="${NODE_ARR[0]}"
    elif [ "$SHUFFLE_RANKS" = true ] && [ $((i % 2)) -eq 1 ]; then
        chunk="${NODE_ARR[1]},${NODE_ARR[0]}"
    else
        chunk="${NODE_ARR[0]},${NODE_ARR[1]}"
    fi

    if [ "$DRY_RUN" = true ]; then
        printf 'repeat %3d %s\n' "$i" "$chunk"
        continue
    fi

    jid=$(CHUNK_ID="$i" TOTAL_CHUNKS="$N_REPEATS" EP="$EP" \
          OUTTAG="$OUTTAG" BENCH_VARIANT="$BENCH_VARIANT" \
          RANKS_PER_NODE="$RANKS_PER_NODE" \
          TIMES="$TIMES" WARMUP="$WARMUP" \
          NUMA_PIN_MODE="$NUMA_PIN_MODE" MPI_NET="$MPI_NET" \
          GATHER_CXI="$GATHER_CXI" GATHER_CXI_COUNTERS_LEVEL="$GATHER_CXI_COUNTERS_LEVEL" \
          sbatch --parsable \
          --uenv=/capstor/scratch/cscs/boeschf/uenv-images/jonathan/openmpi/gh200/store.squashfs \
          --nodes="$N_NODE" \
          --ntasks-per-node="$RANKS_PER_NODE" \
          --cpus-per-task=72 \
          --gpus-per-node=4 \
          --partition=normal \
          --nodelist="$chunk" \
          "$SBATCH_FILE")
    echo "$jid $i $chunk" >>"$MANIFEST"
    printf 'repeat %3d job %s  %s\n' "$i" "$jid" "$chunk"
done

if [ "$DRY_RUN" != true ]; then
    echo
    echo "manifest $MANIFEST"
    echo "watch    squeue -u $USER -n ep-dispatch-mpi-cpu-uenv"
    echo "abort    scancel -u $USER -n ep-dispatch-mpi-cpu-uenv"
    echo "report   python3 $HERE/report_ep_dispatch_custom.py --outdir '${OUTTAG}results-ep-mpi-cpu-uenv-*'"
    if [ "$BENCH_VARIANT" = "nccl" ]; then
        echo "p2p report is not applicable to NCCL variant (raw per-rank data is in per_rank JSON)"
    fi
fi
