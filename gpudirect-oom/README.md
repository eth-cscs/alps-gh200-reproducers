# High memory usage with GPUDirect and unusual allocation patterns

With certain unusual/pathological allocation patterns MPICH/libfabric ends up
using apparently unbounded amounts of GPU memory with GPUDirect. Only
`MPI_Send` (not `MPI_Recv`) seems to be affected.

## Build

With an environment which has `g++`, CUDA, and GPU-aware MPICH loaded:

```bash
g++ -O3 -lcudart -lmpi -lmpi_gtl_cuda gpudirect_oom.cpp -o gpudirect_oom
```

## Run

```bash
MPICH_GPU_SUPPORT_ENABLED=1 srun --gpus-per-task 1 -n 2 gpudirect_oom
```

The test will do an `MPI_Send/Recv` of a large buffer from the first to the
second rank using GPU memory, repeatedly in a loop. Each iteration the program
allocates/deallocates the large buffer that is used for send/recv. The
allocated buffer grows in size each iteration. Additionally, each iteration a
small buffer is allocated which is released at the end of the program.

The program will likely output something like:

```bash
rank: 0, iteration 0
rank: 1, iteration 0
rank: 1, allocated ptr: 0x4002a0000000 of size: 8589934592
rank: 1, allocated ptr: 0x400299e00000 of size: 2097152
rank: 0, allocated ptr: 0x4002a0000000 of size: 8589934592
rank: 0, allocated ptr: 0x400299e00000 of size: 2097152
rank: 0, freeing ptr: 0x4002a0000000
rank: 1, freeing ptr: 0x4002a0000000
rank: 1, cuda_free: 101318262784, cuda_total: 102005473280
rank: 1, iteration 1
rank: 0, cuda_free: 101326192640, cuda_total: 102005473280
rank: 0, iteration 1
...
rank: 0, cuda_free: 15290531840, cuda_total: 102005473280
rank: 0, iteration 11
rank: 1, cuda_free: 101297356800, cuda_total: 102005473280
rank: 1, iteration 11
rank: 0, allocated ptr: 0x4017e0000000 of size: 8613003264
rank: 1, allocated ptr: 0x402d20000000 of size: 8613003264
rank: 0, allocated ptr: 0x4019e1600000 of size: 2097152
rank: 1, allocated ptr: 0x402f21600000 of size: 2097152
rank: 0, freeing ptr: 0x4017e0000000
rank: 1, freeing ptr: 0x402d20000000
rank: 0, cuda_free: 6675431424, cuda_total: 102005473280
rank: 1, cuda_free: 101295259648, cuda_total: 102005473280
rank: 1, iteration 12
rank: 0, iteration 12
CUDA error: 2
rank: 1, allocated ptr: 0x403160000000 of size: 8615100416
rank: 1, allocated ptr: 0x403361800000 of size: 2097152
srun: error: nid006672: task 0: Exited with exit code 1
```

The sending rank (rank 0) reports CUDA error 2, i.e.
`cudaErrorMemoryAllocation`. The GPU runs out of memory. Before the last
iteration the sending rank only has ~6GiB (of 96GiB) of free memory. Each
iteration the free memory is reduced by slightly more than the large allocation
made in the loop.

Not communicating in the loop, i.e. only performing the allocations, only
decreases the free memory by the small allocation size. Running the program
with `MPICH_GPU_IPC_ENABLED=0` to force use of CPU memory for communication
behaves the same as not doing communication.

`compute-sanitizer` does not report any leaks for the program.

# AMD GPUs

The reproducer fails in the same way on AMD GPUs (tested on MI300A). To compile
for AMD GPUs, define `GPUDIRECT_OOM_HIP`, e.g:

```bash
g++ -O3 -DGPUDIRECT_OOM_HIP -D__HIP_PLATFORM_AMD__ -lmpi -lamdhip64 gpudirect_oom.cpp -o gpudirect_oom
```
