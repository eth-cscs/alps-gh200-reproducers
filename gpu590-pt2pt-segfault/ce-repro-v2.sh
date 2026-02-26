#!/bin/bash

cat > ./ce-repro-v2.toml <<EOF
image="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps2"
mounts=["/capstor/scratch/cscs/amadonna/sphexa:/sphexa", "/users/jpcoles/50c.h5:/50c.h5", "$PWD"]
workdir="$PWD"

[env]
OMP_NUM_THREADS = "64"

[annotations]
com.hooks.cxi.enabled = "false"

EOF

cat > ./ce-repro-v2-run.sh <<"EOF"
#!/bin/bash
export LOCAL_RANK=$SLURM_LOCALID
export GLOBAL_RANK=$SLURM_PROCID
export GPUS=(0 1 2 3)
export NUMA_NODE=$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=${GPUS[$NUMA_NODE]}

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MPIIO_HINTS="*:striping_unit=1048576"
 
echo $LOCAL_RANK CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

if [[ $SLURM_PROCID == 0 ]]; then
    echo "compiling..."
    CC=mpicc CXX=mpicxx nvcc -o gpu590 gpu590.cu -g -std=c++17 -L/opt/hpcx/ompi//lib -I//opt/hpcx/ompi/include -Xcompiler "-fPIC" -L/lib/aarch64-linux-gnu/ -lfabric -lcudart  -Xlinker /lib/libfabric.so.1  -Xlinker /usr/lib/libcxi.so.1 -lmpi
fi
sleep 5

echo "running..."
./gpu590
EOF

chmod +x ./ce-repro-v2-run.sh

export SLURM_MPI_TYPE=pmix
export PMIX_MCA_psec=native
srun --ntasks-per-node=2  --environment=./ce-repro-v2.toml ./ce-repro-v2-run.sh

