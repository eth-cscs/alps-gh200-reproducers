#!/bin/bash
#SBATCH --job-name=ep-dispatch-mpi-cpu-uenv
#SBATCH --account=csstaff
# BATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --reservation=SD-70757-apertus-node-diag
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/ep-dispatch-mpi-cpu-uenv-%j.out
#SBATCH --error=slurm_logs/ep-dispatch-mpi-cpu-uenv-%j.err
# BATCH --uenv=prgenv-gnu/26.3:v1
# BATCH --uenv=prgenv-gnu-openmpi/26.3:v1 --view=default
# BATCH --uenv=/capstor/scratch/cscs/boeschf/uenv-images/mojo/store.squashfs
#SBATCH --uenv=/capstor/scratch/cscs/boeschf/uenv-images/jonathan/openmpi/gh200/store.squashfs
# BATCH --uenv=/capstor/scratch/cscs/boeschf/uenv-images/jonathan/cray-mpich/gh200/store.squashfs

set -euo pipefail

# CPU-only MPI EP dispatch benchmark (C binary, supports multiple ranks per node).
# Runs directly in the uenv prgenv-gnu/26.3:v1 with --mpi=cray_shasta.
#
# Usage:
#   sbatch submit_ep_dispatch_mpi_cpu_uenv.sh
#   sbatch --nodelist=nid006106,nid006108,... submit_ep_dispatch_mpi_cpu_uenv.sh
#   sbatch --nodes=1 --ntasks-per-node=4 submit_ep_dispatch_mpi_cpu_uenv.sh  # single-node EP4
#   sbatch --nodes=2 --ntasks-per-node=2 submit_ep_dispatch_mpi_cpu_uenv.sh  # 2-node EP4

export OUTTAG=${OUTTAG:-}
export TIMES=${TIMES:-200}
export WARMUP=${WARMUP:-10}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export OUTDIR="${OUTDIR:-${OUTTAG}results-ep-mpi-cpu-uenv-${SLURM_JOB_ID:-0}}"
export PIN_RANKS=${PIN_RANKS:-""}

export CHUNK_ID="${CHUNK_ID:-0}"
export TOTAL_CHUNKS="${TOTAL_CHUNKS:-1}"

# Optional NUMA pinning mode for each rank.
#   none  - no numactl pinning (default)
#   last  - pin rank to the last CPU of its assigned NUMA node
#   node  - bind rank to all CPUs and memory of its assigned NUMA node
# The assigned node is NUMA_NODE_BASE + LOCAL_RANK.  On GH200 with 72 CPUs per
# NUMA node, NUMA_PIN_MODE=last pins local rank 0 to CPU 71 of node 0, rank 1
# to CPU 143 of node 1, etc.
export NUMA_PIN_MODE="${NUMA_PIN_MODE:-none}"
export NUMA_NODE_BASE="${NUMA_NODE_BASE:-0}"
export NUMA_CPUS_PER_NODE="${NUMA_CPUS_PER_NODE:-72}"

# Optional network transport selection for Cray MPICH / libfabric.
#   cxi     - use Slingshot CXI provider (default)
#   tcp     - force libfabric TCP provider
#   sockets - force libfabric sockets provider
export MPI_NET="${MPI_NET:-cxi}"

# Optional per-rank CXI device pinning. Comma-separated list mapped by
# LOCAL_RANK. Only used when MPI_NET=cxi.
export CXI_DEV_PER_RANK="${CXI_DEV_PER_RANK:-}"

# Optional CXI counter collection. Set GATHER_CXI=1 to wrap each rank with
# gather_cxi_counters. Output lands next to the benchmark JSON.
export GATHER_CXI="${GATHER_CXI:-0}"
export GATHER_CXI_COUNTERS_BIN="${GATHER_CXI_COUNTERS_BIN:-$PWD/gather_cxi_counters}"
export GATHER_CXI_COUNTERS_LEVEL="${GATHER_CXI_COUNTERS_LEVEL:-5}"
export GATHER_CXI_DETAILED="${GATHER_CXI_DETAILED:-0}"
export GATHER_CXI_INTERVAL="${GATHER_CXI_INTERVAL:-100}"

mkdir -p "$OUTDIR" slurm_logs

echo "[$(date --iso-8601=seconds)] job $SLURM_JOB_ID  $SLURM_NNODES nodes  $SLURM_NPROCS ranks  chunk $CHUNK_ID/$TOTAL_CHUNKS"
echo "nodes: $(scontrol show hostlistsorted "$SLURM_JOB_NODELIST")"
export BENCH_VARIANT="${BENCH_VARIANT:-dispatch}"
case "$BENCH_VARIANT" in
    p2p)
        export BENCH_SOURCE="$PWD/ep_dispatch_bench_mpi_cpu_p2p.c"
        export BENCH_BINARY="$PWD/ep_dispatch_bench_mpi_cpu_p2p"
        ;;
    nccl)
        export BENCH_SOURCE="$PWD/ep_dispatch_bench_nccl_p2p.cu"
        export BENCH_BINARY="$PWD/ep_dispatch_bench_nccl_p2p"
        ;;
    dispatch|*)
        export BENCH_SOURCE="$PWD/ep_dispatch_bench_mpi_cpu.c"
        export BENCH_BINARY="$PWD/ep_dispatch_bench_mpi_cpu"
        ;;
esac

# Build the binary if it is missing or the source is newer.
if [ ! -x "$BENCH_BINARY" ] || [ "$BENCH_SOURCE" -nt "$BENCH_BINARY" ]; then
    echo "building $BENCH_SOURCE ..."
    if [ "$BENCH_VARIANT" = "nccl" ]; then
        make -f "$PWD/Makefile.ep_nccl_p2p"
    else
        mpicc -O2 -std=gnu11 -Wall -o "$BENCH_BINARY" "$BENCH_SOURCE"
    fi
    echo "build done"
fi

echo "binary: $BENCH_BINARY"
echo "mpi_net: $MPI_NET"
echo "numa_pin: $NUMA_PIN_MODE"
fi_info --version

LAUNCHER="$PWD/.ep_dispatch_mpi_cpu_uenv_launcher.$SLURM_JOB_ID.sh"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
set -euo pipefail
export LOCAL_RANK=\${SLURM_LOCALID:-0}
export RANK=\$SLURM_PROCID
export WORLD_SIZE=\$SLURM_NPROCS

#if false; then
if command -v ompi_info >/dev/null 2>&1; then

    #export SLURM_CPU_BIND=quiet,mask_cpu:0x000000000000000000000000000000000000000000000000000000FFFFFFFFFFFFF00000,0x000000000000000000000000000000000000FFFFFFFFFFFFFF0000000000000000000000,0x000000000000000000FFFFFFFFFFFFFF0000000000000000000000000000000000000000,0xFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000
    #export FI_CXI_SW_RX_TX_INIT_MAX=\$((8*1024))

    export FI_CXI_RDZV_PROTO=alt_read
    export FI_CXI_RDZV_EAGER_SIZE=0
    export FI_CXI_RDZV_GET_MIN=0
    export FI_CXI_RDZV_THRESHOLD=0

    #unset FI_CXI_RDZV_PROTO
    #unset FI_CXI_RDZV_EAGER_SIZE
    #unset FI_CXI_RDZV_GET_MIN
    #unset FI_CXI_RDZV_THRESHOLD

    export FI_CXI_DEFAULT_CQ_SIZE="131072"
    export FI_CXI_DEFAULT_TX_SIZE="16384"

    #export FI_PROVIDER="^tcp"
    #export FI_PROVIDER="^tcp"

    # Disable registration of host buffers (overflow and request) with GPU.
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
    export FI_CXI_DISABLE_HOST_REGISTER="1"


    # Message matching begins fully offloaded, if resources become exhausted
    # hardware will transition message matching to a hybrid of hardware and
    # software matching, see
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html.
    export FI_CXI_RX_MATCH_MODE=hybrid

    #export FI_CXI_RX_MATCH_MODE=hardware

    # Memory registration cache monitoring method: userfaultfd is required when
    # running applications with NCCL or RCCL.
    export FI_MR_CACHE_MONITOR=userfaultfd

    #export FI_MR_CACHE_MONITOR=kdreg2
    #export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
    #export FI_CXI_RX_MATCH_MODE=software
    #export FI_CXI_CQ_POLICY=always

    # Specify the maximum size and count of memory regions that can be cached. The
    # settings below are used in the HPE Cray Programming Environment (CPE). See,
    # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00006843en_us&page=user/memory_cache_monitor_settings.html
    export FI_MR_CACHE_MAX_SIZE=-1
    export FI_MR_CACHE_MAX_COUNT=524288

    # Defines the maximum CPU memcpy size for HMEM device memory that is accessible
    # by the CPU with load/store operations.
    export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

    export PMIX_MCA_psec="native"

    # The Byte Transfer Layer (BTL) is a framework in Open MPI that is used for
    # point-to-point communication between processes. By setting OMPI_MCA_btl to
    # "^tcp,uct,usnic", we are telling Open MPI to exclude the TCP, UCT, and USNIC
    # BTL components from being used for communication.
    # More information about OMPI_MCA_btl can be found at:
    # https://docs.open-mpi.org/en/v5.0.x/mca.html#frameworks
    export OMPI_MCA_btl="^tcp,uct,usnic"
    export OMPI_MCA_pml="cm"
    export OMPI_MCA_mtl="ofi"
    export OMPI_MCA_opal_common_ofi_provider_include="cxi"

    #export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    #export FI_LNX_PROV_LINKS="shm+cxi:cxi0,cxi1,cxi2,cxi3"
    export OMPI_MCA_mtl_ofi_verbose=200

    #export FI_LNX_SRQ_SUPPORT=1 

    #export FI_OFI_RXM_ENABLE_SHM=1
    #export FI_SHM_USE_XPMEM=1

    #export OMPI_MCA_mtl_ofi_av=table 
    #export OMPI_MCA_btl="^tcp,ofi,vader,openib"
    #export OMPI_MCA_pml="^ucx" 
    #export OMPI_MCA_mtl="ofi" 
    #export OMPI_MCA_opal_common_ofi_provider_include="lnx"

    #export FI_CXI_OFLOW_BUF_SIZE=12582912
    #export FI_CXI_OFLOW_BUF_COUNT=3
    #export FI_CXI_REQ_BUF_SIZE=12582912
    #export FI_CXI_REQ_BUF_MIN_POSTED=6
    #export FI_CXI_REQ_BUF_MAX_CACHED=0
    #export FI_HMEM_ROCR_USE_DMABUF=1
    #export FI_HMEM_CUDA_USE_DMABUF=1


    #export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    #export OMPI_MCA_coll_tuned_alltoall_algorithm=4

    #export FI_CXI_DISABLE_CUDA_SYNC_MEMOPS=1

elif command -v ompi_info >/dev/null 2>&1; then

    unset FI_CXI_RDZV_PROTO
    unset FI_CXI_RDZV_EAGER_SIZE
    unset FI_CXI_RDZV_GET_MIN
    unset FI_CXI_RDZV_THRESHOLD

    export FI_CXI_DEFAULT_CQ_SIZE="131072"
    export FI_CXI_DEFAULT_TX_SIZE="16384"

    # Disable registration of host buffers (overflow and request) with GPU.
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
    export FI_CXI_DISABLE_HOST_REGISTER="1"

    # Message matching begins fully offloaded, if resources become exhausted
    # hardware will transition message matching to a hybrid of hardware and
    # software matching, see
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html.
    export FI_CXI_RX_MATCH_MODE=hybrid

    export FI_CXI_RX_MATCH_MODE=hardware

    # Memory registration cache monitoring method: userfaultfd is required when
    # running applications with NCCL or RCCL.
    export FI_MR_CACHE_MONITOR=userfaultfd

    # Specify the maximum size and count of memory regions that can be cached. The
    # settings below are used in the HPE Cray Programming Environment (CPE). See,
    # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00006843en_us&page=user/memory_cache_monitor_settings.html
    export FI_MR_CACHE_MAX_SIZE=-1
    export FI_MR_CACHE_MAX_COUNT=524288

    # Defines the maximum CPU memcpy size for HMEM device memory that is accessible
    # by the CPU with load/store operations.
    export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

    export PMIX_MCA_psec="native"

    export FI_CXI_DEFAULT_RX_SIZE=1024

    export FI_CXI_OFLOW_BUF_SIZE=12582912
    export FI_CXI_OFLOW_BUF_COUNT=3
    export FI_CXI_REQ_BUF_SIZE=12582912
    export FI_CXI_REQ_BUF_MIN_POSTED=6
    export FI_CXI_REQ_BUF_MAX_CACHED=0
    export FI_HMEM_ROCR_USE_DMABUF=1
    export FI_HMEM_CUDA_USE_DMABUF=1

    export FI_CXI_RDZV_THRESHOLD=16384
    export FI_CXI_RDZV_EAGER_SIZE=2048
    export FI_CXI_SW_RX_TX_INIT_MAX=\$((1*1024))

# -----------------------------------------------------------------------------
# NCCL defaults
# -----------------------------------------------------------------------------

# This forces NCCL to use the libfabric plugin, enabling full use of the
# Slingshot network. If the plugin can not be found, applications will fail to
# start. With the default value, applications would instead fall back to e.g.
# TCP, which would be significantly slower than with the plugin. More
# information about `NCCL_NET` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net
export NCCL_NET="AWS Libfabric"

# Use all interfaces that match "hsn" (the prefix of the Slingshot interfaces,
# hsn = high-speed network) for NCCL's communication. More information about
# `NCCL_SOCKET_IFNAME` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="hsn"

# Allow the use of different NICs for the same ring/tree. This is suited for
# networks where all NICs from a node are connected to the same switch, like
# Slingshot. More information about `NCCL_CROSS_NIC` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cross-nic
export NCCL_CROSS_NIC=0

# Disable inter-node communication using a non-local NIC, using NVLink and an
# intermediate GPU. More information about `NCCL_PXN_DISABLE` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-pxn-disable
export NCCL_PXN_DISABLE=1

# Use P2P when GPUs are connected through NVLink.
# More information about `NCCL_P2P_LEVEL` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level
export NCCL_P2P_LEVEL=NVL

# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. More
# information about `NCCL_NET_GDR_LEVEL` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET_GDR_C2C=1
export NCCL_NET_GDR_READ=1

# Starting with nccl 2.27 a new protocol (LL128) was enabled by default, which
# typically performs worse on Slingshot. The following disables that protocol.
export NCCL_PROTO="^LL128"

# Number of network channels to be used for pairwise communication.
# See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.nChannelsPerNetPeer
export NCCL_NCHANNELS_PER_NET_PEER=4

# Number of CPU helper threads used per network connection for socket
# transport. Increasing this value may increase the socket transport
# performance, at the cost of a higher CPU usage. For generic 100G networks,
# this value can be manually set to 4. However, the product of
# NCCL_SOCKET_NTHREADS and NCCL_NSOCKS_PERTHREAD cannot exceed 64. More
# information about `NCCL_SOCKET_NTHREADS` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-nthreads
export NCCL_SOCKET_NTHREADS=4

# Number of sockets opened by each helper thread of the socket transport. In
# environments where per-socket speed is limited, setting this variable larger
# than 1 may improve the network performance. More information about `NCCL_NSOCKS_PERTHREAD` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nsocks-perthread
export NCCL_NSOCKS_PERTHREAD=1


else

    export FI_CXI_RDZV_PROTO=alt_read
    export FI_CXI_RDZV_EAGER_SIZE=0
    export FI_CXI_RDZV_GET_MIN=0
    export FI_CXI_RDZV_THRESHOLD=0

    unset FI_CXI_RDZV_PROTO
    unset FI_CXI_RDZV_EAGER_SIZE
    unset FI_CXI_RDZV_GET_MIN
    unset FI_CXI_RDZV_THRESHOLD

    export FI_CXI_DEFAULT_CQ_SIZE="131072"
    export FI_CXI_DEFAULT_TX_SIZE="16384"

    # Disable registration of host buffers (overflow and request) with GPU.
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
    export FI_CXI_DISABLE_HOST_REGISTER="1"


    # Message matching begins fully offloaded, if resources become exhausted
    # hardware will transition message matching to a hybrid of hardware and
    # software matching, see
    # https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html.
    export FI_CXI_RX_MATCH_MODE=hybrid

    export FI_CXI_RX_MATCH_MODE=hardware

    export MPICH_ALLTOALL_CHUNKING_MAX_NODES=0
    export MPICH_ALLTOALL_SYNC_FREQ=24


    # Memory registration cache monitoring method: userfaultfd is required when
    # running applications with NCCL or RCCL.
    export FI_MR_CACHE_MONITOR=userfaultfd

    #export FI_MR_CACHE_MONITOR=kdreg2
    #export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
    #export FI_CXI_RX_MATCH_MODE=software
    #export FI_CXI_CQ_POLICY=always

    # Specify the maximum size and count of memory regions that can be cached. The
    # settings below are used in the HPE Cray Programming Environment (CPE). See,
    # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00006843en_us&page=user/memory_cache_monitor_settings.html
    export FI_MR_CACHE_MAX_SIZE=-1
    export FI_MR_CACHE_MAX_COUNT=524288

    # Defines the maximum CPU memcpy size for HMEM device memory that is accessible
    # by the CPU with load/store operations.
    export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

    export MPICH_GPU_SUPPORT_ENABLED=0
    #export MPICH_OFI_VERBOSE=1
    #export MPICH_OFI_DEFAULT_TCLASS=TC_LOW_LATENCY
    #export MPICH_OFI_TCLASS_ERRORS=ERROR

    export PMIX_MCA_psec="native"

    #export FI_CXI_DISABLE_ALT_READ_CMDQ=1


    export MPICH_RANK_REORDER_DISPLAY=1
    export MPICH_OFI_NIC_VERBOSE=2


fi

# Cray MPICH / libfabric transport selection.
case "\${MPI_NET:-cxi}" in
    tcp)
        export MPICH_OFI_USE_PROVIDER="tcp;ofi_rxm"
        export MPICH_OFI_ENABLE_HMEM=0
        export MPICH_OFI_STARTUP_CONNECT=1
        ;;
    sockets)
        export MPICH_OFI_USE_PROVIDER=sockets
        export MPICH_OFI_ENABLE_HMEM=0
        ;;
    cxi|*)
        # Default Slingshot/CXI path.
        ;;
esac

# Per-rank CXI device selection on the CXI path.
if [ "\${MPI_NET:-cxi}" = "cxi" ] && [ -n "\${CXI_DEV_PER_RANK:-}" ]; then
    IFS=',' read -ra CXI_DEVS <<< "\$CXI_DEV_PER_RANK"
    if [ "\$LOCAL_RANK" -lt "\${#CXI_DEVS[@]}" ]; then
        export FI_CXI_DEVICE_NAME="\${CXI_DEVS[\$LOCAL_RANK]}"
    else
        echo "warning: LOCAL_RANK=\$LOCAL_RANK out of range for CXI_DEV_PER_RANK" >&2
    fi
fi

# NUMA pinning.
NUMA_PREFIX=()
case "\${NUMA_PIN_MODE:-none}" in
    last)
        _NODE=\$((NUMA_NODE_BASE + \${SLURM_LOCALID:-0}))
        _LAST_CPU=\$(( (_NODE + 1) * NUMA_CPUS_PER_NODE - 1 ))
        NUMA_PREFIX=(numactl --physcpubind="\$_LAST_CPU" --membind="\$_NODE")
        ;;
    net)
        _NODE=\$((NUMA_NODE_BASE + \${SLURM_LOCALID:-0}))
        NUMA_PREFIX=(numactl --cpunodebind=netdev:hsn"\$_NODE" --membind=netdev:hsn"\$_NODE")
        ;;
    node)
        _NODE=\$((NUMA_NODE_BASE + \${SLURM_LOCALID:-0}))
        NUMA_PREFIX=(numactl --cpunodebind="\$_NODE" --membind="\$_NODE")
        ;;
    none|*)
        NUMA_PREFIX=()
        ;;
esac

BENCH="$BENCH_BINARY"
BENCH_ARGS=(
    --ep "\$SLURM_NPROCS"
    --iters "$TIMES"
    --warmup "$WARMUP"
    --chunk-id "$CHUNK_ID"
    --total-chunks "$TOTAL_CHUNKS"
    --balanced-routing
    --pin "$PIN_RANKS"
    --outdir "$OUTDIR"
    --out "$OUTDIR/ep-dispatch-\$SLURM_JOB_ID.json"
    "\$@"
)

if [ -n "\${GATHER_CXI:-}" ] && [ "\${GATHER_CXI}" != "0" ]; then
    export GATHER_CXI_COUNTERS_LEVEL="\${GATHER_CXI_COUNTERS_LEVEL:-5}"
    export GATHER_CXI_DETAILED="\${GATHER_CXI_DETAILED:-0}"
    export GATHER_CXI_INTERVAL="\${GATHER_CXI_INTERVAL:-100}"
    export GATHER_CXI_JSON=1
    export GATHER_CXI_OS_METRICS=0
    CTR_TAG="pair-\$SLURM_JOB_ID-r\$RANK"
    CTR_OUT="$OUTDIR/cxi-counters-\$SLURM_JOB_ID-rank\$RANK.json"
    echo "\${NUMA_PREFIX[@]}" "$GATHER_CXI_COUNTERS_BIN" -e "\$CTR_TAG" "\$BENCH" "\${BENCH_ARGS[@]}"
    if [ "\${SLURM_LOCALID:-0}" -eq 0 ]; then
        "\${NUMA_PREFIX[@]}" "$GATHER_CXI_COUNTERS_BIN" -e "\$CTR_TAG" "\$BENCH" "\${BENCH_ARGS[@]}" > "\$CTR_OUT" 2> "\$CTR_OUT.err"
    else
        "\${NUMA_PREFIX[@]}" "$GATHER_CXI_COUNTERS_BIN" -e "\$CTR_TAG" "\$BENCH" "\${BENCH_ARGS[@]}" > /dev/null 2> "\$CTR_OUT.err"
    fi
else
    echo "\${NUMA_PREFIX[@]}" "\$BENCH" "\${BENCH_ARGS[@]}"
    "\${NUMA_PREFIX[@]}" "\$BENCH" "\${BENCH_ARGS[@]}"
fi
EOF
chmod +x "$LAUNCHER"

cleanup() { rm -f "$LAUNCHER"; }
trap cleanup EXIT

export TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
export CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-72}"

     #--distribution=block:block \
#if false; then
if command -v ompi_info >/dev/null 2>&1; then
srun -l --mpi=pmix \
    --mem-bind=local \
     --network=disable_rdzv_get \
     --ntasks-per-node="$TASKS_PER_NODE" \
     --cpus-per-task="$CPUS_PER_TASK" \
     "$LAUNCHER" "$@" --no-p2p
elif command -v ompi_info >/dev/null 2>&1; then
srun --mpi=pmix \
     --ntasks-per-node="$TASKS_PER_NODE" \
     --cpus-per-task="$CPUS_PER_TASK" \
     "$LAUNCHER" "$@" --no-p2p
else

     #--network=disable_rdzv_get \
srun --mpi=cray_shasta \
     --ntasks-per-node="$TASKS_PER_NODE" \
     --cpus-per-task="$CPUS_PER_TASK" \
     --distribution=block:block \
     "$LAUNCHER" "$@"

fi
