#!/bin/bash
#SBATCH --nodes=3
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=elbencho-netbench
#SBATCH --time=02:00:00

SRUN="srun"
ELBENCHO="./elbencho"

# Comma-separated list of allocated node names
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
read -r SERVER CLIENT1 CLIENT2 <<< \
  "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

export SRUN ELBENCHO SERVER CLIENT1 CLIENT2

#set -euo pipefail

function run_test()
{
    DEVS=$1 ; shift
    THREADS=$1 ; shift
    # Remaining arguments are individual client hostnames
    CLIENTS=$(IFS=,; echo "$*")
    NUM_CLIENTS=$#
    LABEL="$DEVS,$THREADS"

    # Start one service on every allocated node (server + each client)
    $SRUN -l -w "$SERVER,$CLIENTS" --ntasks=$((NUM_CLIENTS + 1)) --ntasks-per-node=1 $ELBENCHO --service --foreground --netdevs $DEVS &
    SERVICE_STEP=$!

    cleanup() {
        $ELBENCHO --quit --hosts "$HOSTS" || true
        kill "$SERVICE_STEP" 2>/dev/null || true
    }
    trap cleanup EXIT

    $ELBENCHO --label "$LABEL" --resfile results-${SLURM_JOBID}.txt \
              --netprov cxi --netengine libfabric --netbench \
              --servers "$SERVER" --clients "$CLIENTS" -t $THREADS \
              --sendsize 1M --recvsize 0 \
              --timelimit 150 --base10 --graph --live1n
    wait
}
export -f run_test

function one_test()
{
    timeout 3m bash -c "run_test $*"
    #sleep 10
}

one_test cxi[0-3] 1 "$CLIENT1" 
one_test cxi[0-3] 2 "$CLIENT1" 
one_test cxi[0-3] 4 "$CLIENT1" 
one_test cxi[0-3] 8 "$CLIENT1" 
one_test cxi[0-3] 16 "$CLIENT1" 
one_test cxi[0-3] 32 "$CLIENT1" 
one_test cxi[0-1] 32 "$CLIENT1" 
one_test cxi[2-3] 32 "$CLIENT1" 
exit

one_test cxi[0-3] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[0-3] 16 "$CLIENT1" "$CLIENT2"
one_test cxi[0-3] 8 "$CLIENT1" "$CLIENT2"
one_test cxi[0-3] 4 "$CLIENT1" "$CLIENT2"
one_test cxi[0-3] 2 "$CLIENT1" "$CLIENT2"
one_test cxi[0-3] 1 "$CLIENT1" "$CLIENT2"

one_test cxi[0-1] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[2-3] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[0] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[1] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[2] 32 "$CLIENT1" "$CLIENT2"
one_test cxi[3] 32 "$CLIENT1" "$CLIENT2"

