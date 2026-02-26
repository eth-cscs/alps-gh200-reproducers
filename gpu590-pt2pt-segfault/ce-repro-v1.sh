#!/bin/bash

cat > ./ce-repro-v1.toml <<EOF
image="quay.io/ethcscs/sph-exa:0.95-mpich4.3.1-ofi1.22-cuda12.8"
mounts=["$PWD"]
workdir="$PWD"

[env]
OMP_NUM_THREADS = "64"

[annotations]
com.hooks.cxi.enabled = "false"
EOF

cat > ./ce-repro-v1-run.sh <<"EOF"
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
    #CC=mpicc CXX=mpicxx nvcc -o gpu590 gpu590.cu -g -std=c++17 -Xcompiler "-fPIC" -L/lib/aarch64-linux-gnu/ -lcudart -lmpi -Xlinker /usr/lib64/libcxi.so.1 -Xlinker /usr/lib64/libnl-3.so.200
    CC=mpicc CXX=mpicxx nvcc -o gpu590 gpu590.cu -g -std=c++17 -Xcompiler "-fPIC" -L/lib/aarch64-linux-gnu/ -lcudart -lmpi 
fi
sleep 5

echo "running..."
#./gpu590
gdb -batch \
    -ex 'run' \
    -ex 'bt' \
    -ex 'quit' \
    --args ./gpu590
EOF

chmod +x ./ce-repro-v1-run.sh

srun --mpi=pmi2 -N1 --ntasks-per-node=2  --environment=./ce-repro-v1.toml ./ce-repro-v1-run.sh

