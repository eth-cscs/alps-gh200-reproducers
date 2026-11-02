#!/bin/bash
#SBATCH --nodes=3
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=elbencho-netbench
#SBATCH --time=00:05:00

SRUN="srun"

set -euo pipefail

#export FI_LOG_LEVEL=info 
#export FI_LOG_PROV=cxi
#export FI_LOG_SUBSYS=ep_data

# Comma-separated list of allocated node names
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
read -r SERVER CLIENT1 CLIENT2 <<< \
  "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

# Start one service on every allocated node
$SRUN --ntasks=3 --ntasks-per-node=1 ./elbencho --service --foreground --netdevs cxi[0-3] &
#$SRUN --ntasks=3 --ntasks-per-node=1 ./elbencho --service --foreground --netdevs cxi3 &
SERVICE_STEP=$!

cleanup() {
    ./elbencho --quit --hosts "$HOSTS" || true
    kill "$SERVICE_STEP" 2>/dev/null || true
}
trap cleanup EXIT

# Give the services time to start, then run 2 clients against 1 server
sleep 3

./elbencho --netbench --servers "$HOSTS" --clients "$HOSTS" -t 32 --sendsize 1M --recvsize 0 --timelimit 40 --base10 --graph

