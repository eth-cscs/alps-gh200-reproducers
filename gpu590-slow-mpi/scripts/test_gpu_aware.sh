#!/bin/bash

for state in 0 1
do
    export MPICH_GPU_SUPPORT_ENABLED=$state
    echo
    echo "=========== MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED"
    echo
    srun -n32 -c1 -N1 -Acsstaff ./pdsygst
done
