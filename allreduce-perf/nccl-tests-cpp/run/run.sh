#!/bin/bash

NODES=$1
NTASKS_PER_NODE=$2
TIME_LIMIT=${3:-"01:00:00"}

CPUS_PER_TASK=$(( 288 / NTASKS_PER_NODE ))
NTASKS=$(( NODES * NTASKS_PER_NODE ))

NODES_STR=$(printf "%04d" "$NODES")
NTASKS_STR=$(printf "%05d" "$NTASKS")

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name allreduce_bench
#SBATCH --output=job_n_${NTASKS_STR}_N_${NODES_STR}_TPN_${NTASKS_PER_NODE}.out
#SBATCH --error=job_n_${NTASKS_STR}_N_${NODES_STR}_TPN_${NTASKS_PER_NODE}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --exclusive

export OMP_NUM_THREADS=${CPUS_PER_TASK}
export NCCL_DEBUG=INFO

check_health() {
    local srunlist=job_n_${NTASKS_STR}_N_${NODES_STR}_TPN_${NTASKS_PER_NODE}.srunlist

    srun --cpu-bind=verbose,rank_ldom --network=single_node_vni hostname > \$srunlist

    local unique_nodes=\$(sort \$srunlist | uniq | wc -l)
    local total_tasks=\$(wc -l < \$srunlist)

    if [ "\$unique_nodes" -ne "$NODES" ] || [ "\$total_tasks" -ne "$NTASKS" ]; then
        echo "One or more nodes or tasks are faulty. Exiting."
        exit 1
    else
        echo "All nodes and tasks are healthy."
    fi
}

check_health

srun \
    --cpu-bind=verbose,rank_ldom \
    --environment="$(realpath nccl-test.toml)" \
    --mpi=pmi2 \
    all_reduce_perf -b 4294967296 -e 4294967296 -g 1

EOT
