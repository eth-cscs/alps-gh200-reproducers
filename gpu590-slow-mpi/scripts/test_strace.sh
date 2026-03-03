#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export GOTO_NUM_THREADS=1

if [ $SLURM_PROCID -eq 0 ]; then
    strace -c -o strace.$CLUSTER_NAME.out ./pdsygst
    cat strace.$CLUSTER_NAME.out
else
    ./pdsygst
fi

