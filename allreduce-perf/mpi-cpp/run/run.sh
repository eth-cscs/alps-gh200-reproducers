#!/bin/bash

NODES=$1
NTASKS_PER_NODE=$2
TIME_LIMIT=${3:-"01:00:00"}

CPUS_PER_TASK=$(( 288 / NTASKS_PER_NODE ))
NTASKS=$(( NODES * NTASKS_PER_NODE ))

NODES_STR=$(printf "%04d" "$NODES")
NTASKS_STR=$(printf "%05d" "$NTASKS")

sbatch \
    --uenv=/bret/scratch/cscs/boeschf/alps-gh200-reproducers/allreduce-perf/mpi-cpp/env/store.squashfs <<EOT
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

uenv view default
export LD_LIBRARY_PATH=/user-environment/env/default/lib64:\$LD_LIBRARY_PATH

ITERATIONS=10
SUB_ITERATIONS=5
SIZE=$((1 << 30))

run_all() {
    #srun \
    #    --cpu-bind=rank_ldom \
    #    ../../common/launch_wrapper \
    #    ../src/build/allreduce_bench \$ITERATIONS \$SUB_ITERATIONS host \$SIZE \$1 \$2

    #srun \
    #    --cpu-bind=rank_ldom \
    #    ../../common/launch_wrapper \
    #    ../src/build/allreduce_bench \$ITERATIONS \$SUB_ITERATIONS pinned_host \$SIZE \$1 \$2

    #srun \
    #    --cpu-bind=rank_ldom \
    #    ../../common/launch_wrapper \
    #    ../src/build/allreduce_bench \$ITERATIONS \$SUB_ITERATIONS mpi_host \$SIZE \$1 \$2

    srun \
        --cpu-bind=rank_ldom \
        ../../common/launch_wrapper \
        ../src/build/allreduce_bench \$ITERATIONS \$SUB_ITERATIONS device \$SIZE \$1 \$2
}

run_all 0 0
run_all 1 0
run_all 0 1
run_all 1 1

EOT

