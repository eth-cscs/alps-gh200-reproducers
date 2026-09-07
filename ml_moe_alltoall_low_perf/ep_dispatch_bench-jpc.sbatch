#!/bin/bash
# EP dispatch latency benchmark: N nodes x 4 GPUs = 4N ranks = one EP group.
# Defaults to EP8 (2 nodes); EP and --nodes must be changed together.
#
#   sbatch ep_dispatch_bench.sbatch                       # 2 nodes anywhere
#   EP=16 sbatch --nodes=4 ep_dispatch_bench.sbatch       # one EP16 group
#   EP=32 sbatch --nodes=8 ep_dispatch_bench.sbatch       # one EP32 group
#   sbatch --nodelist=nid00[6570-6571] ep_dispatch_bench.sbatch
#   sbatch --exclude=/capstor/.../common/filter/exclude_all.txt ep_dispatch_bench.sbatch
#   sbatch --nodes=64 ep_dispatch_bench.sbatch            # 32 EP8 groups at once
#   sbatch ep_dispatch_bench.sbatch --backend nccl        # all_to_all_single, no UCCL
#
# Trailing args go to the python program: sbatch ... bench/ep_dispatch_bench.sbatch --fp8 --iters 100
# out/err land in logs/ (absolute: #SBATCH cannot expand a variable, and the
# dir must already exist); the result JSON goes to logs/ep-bench-2n[-BACKEND].

#SBATCH --account=csstaff
#SBATCH --job-name=bench_bad_nodes
#SBATCH --time=0:5:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue
# BATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --reservation=SD-70757-apertus-node-diag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SETFREQ=${SETFREQ:-0}

IMAGE_ENV=./alps-pytorch2512.toml
IMAGE_ENV=./alps-pytorch2602.toml
source ./paths.sh

# Both are read by common/uccl.sh to pick the tuned chunk configs, so the config
# tracks the payload dtype the python program was asked for.
# EP must match the ranks the job actually launches (4 per node), so override it
# together with --nodes -- bench_all.sh passes both. The #SBATCH --nodes
# default above is the matching one for this default.
EP="${EP:-8}"
USE_FP8_DISPATCH=false
[[ " $* " == *" --fp8 "* ]] && USE_FP8_DISPATCH=true
# source "$SCRIPTS_ROOT/common/uccl.sh"

# --backend nccl needs neither the UCCL build nor its chunk configs.
BACKEND=uccl
case " $* " in *" --backend nccl "* | *" --backend=nccl "*) BACKEND=nccl ;; esac
# 
# if [ "$BACKEND" = uccl ] && [ ! -d "$UCCL_INSTALL_TARGET" ]; then
# 	echo "no UCCL build at $UCCL_INSTALL_TARGET -- build it once with install_uccl (common/uccl.sh)" >&2
# 	exit 1
# fi

# Its own dir: the report tools pool a whole directory, and a NCCL latency is
# not comparable to a UCCL one.
RESULT_DIR="./logs/ep-bench-2n"
if [ "$BACKEND" != uccl ]; then
	RESULT_DIR="$RESULT_DIR-$BACKEND"
fi
mkdir -p "$RESULT_DIR"
RESULT="$RESULT_DIR/ep-dispatch-$SLURM_JOB_ID.json"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=25679
export WORLD_SIZE=$SLURM_NPROCS
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / ${SLURM_GPUS_PER_NODE:-4}))

#export NCCL_BUFFSIZE=16777216         # 16MB buffer (up from 4MB)
#export NCCL_ALGO=Ring                 # explicit ring for 2-8 node clusters
#export NCCL_PROTO=Simple              # Simple protocol for IB inter-node
#export NCCL_NTHREADS=512


#
#export OMP_NUM_THREADS=1
#export OMP_WAIT_POLICY=PASSIVE
#export NCCL_SOCKET_NTHREADS=8
#export NCCL_NSOCKS_PERTHREAD=8

#export NCCL_DEBUG=info

#export NCCL_GRAPH_MIXING_SUPPORT=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Deliberately NOT setting CUDA_DEVICE_MAX_CONNECTIONS=1 as train.sh does: that
# exists for Megatron's TP overlap and would only serialise streams here.

echo "[$(date --iso-8601=seconds)] job $SLURM_JOB_ID  $SLURM_NNODES nodes  $WORLD_SIZE ranks"
echo "nodes: $(scontrol show hostlistsorted "$SLURM_JOB_NODELIST")"
echo "backend $BACKEND  dispatch cfg ${UCCL_EP_DISPATCH_CONFIG:-uccl-dispatch-config-not-set}   combine cfg ${UCCL_EP_COMBINE_CONFIG:-uccl-combine-config-not-set}"

rank_masks=$(
  python3 - <<'PY'
n=72-16
#for start in (36, 108, 180, 252):
#for start in (70, 70+72*1, 70+72*2, 70+72*3):
#for start in (70, 70+72*0, 70+72*0, 70+72*0):
for start in (72-n, (72-n)+72*1, (72-n)+72*2, (72-n)+72*3):
    mask = ((1 << n) - 1) << start
    print(f'{mask:x}', end=',')
PY
)
rank_masks=${rank_masks%,}

        #--cpu-bind=cores \

echo "image: $IMAGE_ENV"

if [[ $SETFREQ == 1 ]]; then
    echo "-- Setting Frequency --"
    srun bash -c "sudo cpupower frequency-set -g ondemand"
    srun bash -c "echo 3000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"
    srun bash -c "echo 3000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"
    srun -l bash -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort -u"
    srun -l bash -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq | sort -u"
    echo "-- Finished Setting Frequency --"
fi

# --wait/--kill-on-bad-exit: one dead rank must not leave the rest spinning.
# RANKMAP mirrors train.sh so placement can be recovered from the log by grep.
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" \
	--mpi=pmix \
	--environment="$IMAGE_ENV" \
	--wait 60 --kill-on-bad-exit=1 -lu \
	--distribution=block:block \
	--network=disable_rdzv_get \
        --cpu-bind=verbose,mask_cpu:${rank_masks} \
        bash -c "
unset NCCL_CROSS_NIC
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NETDEVS_POLICY=MAX:1
export NCCL_P2P_LEVEL=PHB
export NCCL_GDRCOPY_ENABLE=1

#export NCCL_NET_SHARED_COMMS=0
#export NCCL_GDRCOPY_FLUSH_ENABLE=1

export NCCL_NVB_DISABLE=1

unset NCCL_PXN_DISABLE
export NCCL_P2P_PXN_LEVEL=0

#export NCCL_PROTO=SIMPLE
#export NCCL_P2P_NET_CHUNKSIZE=\$((4*1024*1024))
#
#export NCCL_IPC_USE_ABSTRACT_SOCKET=1

# export OMP_PROC_BIND=true; 
# export OMP_NUM_THREADS=$((72-16-8)); 
# export OMP_NUM_THREADS=$((72-16-8-24)); 
# #export OMP_PLACES=cores; 
# #export OMP_PLACES={\$((\$SLURM_LOCALID * 72 + 24)):\$OMP_NUM_THREADS:1}; 
# export OMP_PLACES={\$((\$SLURM_LOCALID * 72 + 72-\$OMP_NUM_THREADS)):\$OMP_NUM_THREADS:1}; 
# export OMP_DISPLAY_AFFINITY=true; 
# export OMP_DISPLAY_ENV=verbose; 
# export OMP_WAIT_POLICY=ACTIVE;
                numactl -N \$SLURM_LOCALID -m \$SLURM_LOCALID numactl -s; 
        	echo RANKMAP \$SLURM_PROCID \$SLURMD_NODENAME; 
                HSN=(hsn0 hsn2 hsn3 hsn1); 
                device=\${HSN[\$SLURM_LOCALID]}; 
                echo Local rank \$SLURM_LOCALID  \$device; 
                export NCCL_SOCKET_IFNAME=\$device; 
        	RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        	python3 ./ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"
        	#./bench/log_cooling_wrapper.sh python3 ./bench/ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"
        	#numactl -N \$SLURM_LOCALID -m \$SLURM_LOCALID python3 ./bench/ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"
        	#numactl --physcpubind=+64-71 python3 ./bench/ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"
        	#python3 ./bench/ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"
#       bash -c "export PYTHONPATH=$UCCL_PYTHONPATH:\${PYTHONPATH:-}; \
#       	echo RANKMAP \$SLURM_PROCID \$SLURMD_NODENAME; \
#       	RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
#               nsys profile \
#                 --trace=cuda,nvtx,osrt,python-gil \
#                 --python-sampling=true \
#                 --sample=cpu \
#                 --cpuctxsw=process-tree \
#                 --event-sample=system-wide \
#                 --output=profiles/train-%q{SLURM_PROCID}-%h \
#                 --force-overwrite true \
#       	numactl --physcpubind=+64-71 python3 ./bench/ep_dispatch_bench-jpc.py --ep $EP --out $RESULT $*"

if [[ $SETFREQ == 1 ]]; then
    echo "-- Resetting Frequency --"
    srun bash -c "sudo cpupower frequency-set -g performance"
    srun bash -c "echo 4000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"
    srun bash -c "echo   80000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"
    srun -l bash -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort -u"
    srun -l bash -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq | sort -u"
    echo "-- Finished Resetting Frequency --"
fi
