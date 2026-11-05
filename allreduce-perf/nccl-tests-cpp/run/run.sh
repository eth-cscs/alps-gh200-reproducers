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

set -e

unset UENV_MOUNT_LIST
export OMP_NUM_THREADS=${CPUS_PER_TASK}
export NCCL_DEBUG=INFO

check_gpus() {

    local srunlist=job_n_${NTASKS_STR}_N_${NODES_STR}_TPN_${NTASKS_PER_NODE}.srunlist
    srun -l --cpu-bind=verbose,rank_ldom --network=single_node_vni hostname > \$srunlist

    local mem_state=job_n_${NTASKS_STR}_N_${NODES_STR}_TPN_${NTASKS_PER_NODE}.mem
    srun -l --cpu-bind=rank_ldom nvidia-smi --query-gpu=index,uuid,memory.free,ecc.errors.uncorrected.volatile.total --format=csv,nounits,noheader > \${mem_state}

    local min_free_mem=80000

    # Read the node information into an associative array
    declare -A node_map
    while IFS=: read -r rank node; do
        node_map[\$rank]=\$node
    done < "\$srunlist"

    # Count unique values in node_map
    local unique_values=()
    for value in "\${node_map[@]}"; do
        if [[ ! " \${unique_values[@]} " =~ " \$value " ]]; then
            unique_values+=("\$value")
        fi
    done

    local num_entries=\${#node_map[@]}
    local num_unique=\${#unique_values[@]}
    if [ "\$num_unique" -ne "$NODES" ] || [ "\$num_entries" -ne "$NTASKS" ]; then
        echo "One or more nodes or tasks are faulty."
        return 1
    else
        echo "All nodes and tasks are healthy."
    fi

    # Process the GPU info file and check conditions
    local failure=false
    while IFS=: read -r rank gpu_info; do
        # Split the GPU info into fields
        IFS=',' read -r gpu_id gpu_name free_mem error_indicator <<< "\$gpu_info"

        # Trim leading and trailing whitespace from fields
        gpu_id=\$(echo "\$gpu_id" | xargs)
        gpu_name=\$(echo "\$gpu_name" | xargs)
        free_mem=\$(echo "\$free_mem" | xargs)
        error_indicator=\$(echo "\$error_indicator" | xargs)

        # Check conditions
        if (( free_mem <= min_free_mem || error_indicator != 0 )); then
            echo "Failing rank: \$rank, Node name: \${node_map[\$rank]}, GPU ID: \$gpu_id, GPU name: \$gpu_name, Free memory: \$free_mem, Error indicator: \$error_indicator"
            failure=true
        fi
    done < "\$mem_state"

    if \$failure; then
        echo "One or more gpus are faulty."
        return 1
    else
        echo "All ranks passed the checks."
        return 0
    fi
}

check_gpus

unset UENV_MOUNT_LIST

srun \
    --cpu-bind=verbose,rank_ldom \
    --environment="$(realpath nccl-test.toml)" \
    --mpi=pmi2 \
    all_reduce_perf -b 4294967296 -e 4294967296 -g 1

EOT
