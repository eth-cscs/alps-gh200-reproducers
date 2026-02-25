#!/bin/bash

if [[ $CLUSTER_NAME == "daint" ]]; then
    export SLURM_PARTITION="debug"
fi

if [[ $CLUSTER_NAME == "clariden" ]]; then
    export SLURM_PARTITION="debug"
fi


function debug()
{
    gdb -batch \
        -ex 'run' \
        -ex 'bt' \
        -ex 'quit' \
        --args "$@" #> /users/jpcoles/gpu590-gdb.$LOCAL_RANK.out 2>&1
}
export -f debug

function main()
{
    #export MPICH_OPTIMIZED_MEMCPY=0
    #export MPICH_GPU_SUPPORT_ENABLED=0

    set -x
#    export FI_LOG_LEVEL=info
#    export FI_LOG_SUBSYS=mr

    srun osu_alltoall H H
    srun osu_alltoall D D
    srun osu_bw H H
    srun osu_bw H D
    srun osu_bw D H
    srun osu_bw D D
    srun bash -c "debug osu_bw D D"
}
export -f main

uenv image pull prgenv-gnu/25.11:v1
uenv image pull prgenv-gnu-openmpi/25.12:v1@daint
uenv image pull service::prgenv-gnu/25.11:test1@starlex

uenv start --ignore-tty prgenv-gnu/25.11:test1 --view=default <<EOF
salloc -A csstaff -t 5:00 -N1 -n2 bash -c "main"
salloc -A csstaff -t 5:00 -N2 -n2 bash -c "main"
EOF

exit

uenv start --ignore-tty prgenv-gnu/25.11:v1 --view=default <<EOF
salloc -A csstaff -t 5:00 -N1 -n2 bash -c "main"
salloc -A csstaff -t 5:00 -N2 -n2 bash -c "main"
EOF

uenv start --ignore-tty prgenv-gnu-openmpi/25.12:v1@daint --view=default <<EOF
export SLURM_MPI_TYPE=pmix
export PMIX_MCA_psec=native
salloc -A csstaff -t 5:00 -N1 -n2 bash -c "main"
EOF
