#!/bin/bash
#SBATCH --nodes=3
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=elbencho-netbench
#SBATCH --time=02:00:00

SRUN="srun"
ELBENCHO="./elbencho"
export SRUN ELBENCHO

set -euo pipefail

function run_test()
{
    DEVS=${1:-cxi[0-3]}
    THREADS=${2:-32}

    # Comma-separated list of allocated node names
    HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
    read -r SERVER CLIENT1 CLIENT2 <<< \
      "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

    # Start one service on every allocated node
    $SRUN -l --ntasks=3 --ntasks-per-node=1 $ELBENCHO --service --foreground --netdevs $DEVS &
    SERVICE_STEP=$!

    cleanup() {
        $ELBENCHO --quit --hosts "$HOSTS" || true
        kill "$SERVICE_STEP" 2>/dev/null || true
    }
    trap cleanup EXIT

    # Give the services time to start, then run 2 clients against 1 server
    sleep 3

    LABEL="$DEVS,$THREADS"

    $ELBENCHO --label "$LABEL" --resfile results.txt \
              --netprov cxi --netengine libfabric --netbench \
              --servers "$SERVER" --clients "$CLIENT1,$CLIENT2" -t $THREADS \
              --sendsize 1M --recvsize 0 \
              --timelimit 300 --base10 --graph --live1n
    wait
}
export -f run_test

function one_test()
{
    timeout 6m bash -c "run_test $@"
    sleep 10
}

one_test cxi[0-3] 32
one_test cxi[0-3] 16
one_test cxi[0-3] 8
one_test cxi[0-3] 4
one_test cxi[0-3] 2
one_test cxi[0-3] 1

one_test cxi[0-1] 32
one_test cxi[2-3] 32
one_test cxi[0] 32
one_test cxi[1] 32
one_test cxi[2] 32
one_test cxi[3] 32

