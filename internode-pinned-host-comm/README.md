# Slow communication from pinned host memory within the node

Communication from pinned host memory can be significantly slower than going
over the NIC (either within a node or across nodes).

## Build

With an environment which has `g++` and CUDA loaded (if GPU-aware MPI, include
`mpi_gtl_cuda`, otherwise leave it out):

```bash
g++ -lcudart -lmpi -lmpi_gtl_cuda internode_pinned_host_comm.cpp -o internode_pinned_host_comm
```

## Run

```bash
srun -n 2 internode_pinned_host_comm <num_iterations> <num_sub_iterations> <mem_type> <p2p_size>
```

The test will do an `MPI_Send/Recv` from the first to the last rank of
`<p2p_size>` bytes from a `<mem_type>` buffer. `<mem_type>` must be one of
`host`, `pinned_host` (allocated using `cudaMallocHost`), or `device`
(allocated using `cudaMalloc`). The test will perform one warmup iteration, and
then `<num_iterations>` allocations with communication. Each iteration performs
`<num_sub_iterations>` `MPI_Send/Recv`s with the same allocation.

The tests below were made with `MPICH_SMP_SINGLE_COPY_MODE=CMA` since
`MPICH_GPU_SUPPORT_ENABLED=1` disables use of xpmem. However, performance seems
to be almost identical between cma and xpmem.

To communicate 128MiB of host-pinned memory within a node:

```bash
srun -n 2 -N 1 --cpu-bind=sockets internode_pinned_host_comm 2 5 pinned_host $((1 << 27))
```

The program will likely output something like:

```bash
mem_type: pinned_host
p2p_size: 134217728
[-1] time: 3.97116
Doing MPI_Send/Recv from rank 0 to rank 1, of 134217728 bytes
[0:0] time: 0.0246628
[0:1] time: 0.020258
[0:2] time: 0.0187662
[0:3] time: 0.0189267
[0:4] time: 0.0190948
[1:0] time: 0.0243361
[1:1] time: 0.0200991
[1:2] time: 0.0186949
[1:3] time: 0.0186447
[1:4] time: 0.0189611
```

Communicating via the NIC within a node:

```bash
MPIR_CVAR_NO_LOCAL=1 srun -n 2 -N 1 --cpu-bind=sockets internode_pinned_host_comm 2 5 pinned_host $((1 << 27))
```

will likely result in something like:

```bash
PE 0: MPICH Warning: MPICH_NO_LOCAL is set to 1.
      This setting disables all intra-node MPI optimizations.
      In addition, on systems with multiple NICs per node, this setting
      forces HPE Cray MPI to only use a single NIC on each node. This
      setting is intended only for debugging and may significantly
      impact MPI performance.

mem_type: pinned_host
p2p_size: 134217728
[-1] time: 0.0153468
Doing MPI_Send/Recv from rank 0 to rank 1, of 134217728 bytes
[0:0] time: 0.0104991
[0:1] time: 0.00576195
[0:2] time: 0.00576105
[0:3] time: 0.00576077
[0:4] time: 0.00576077
[1:0] time: 0.00912396
[1:1] time: 0.00576182
[1:2] time: 0.00576217
[1:3] time: 0.00575993
[1:4] time: 0.00575907
```

After the first iteration per allocation, the communication is roughly four
times faster than when using CMA.

Communicating via the NIC with ranks across nodes:

```bash
srun -n 2 -N 2 --cpu-bind=sockets internode_pinned_host_comm 2 5 pinned_host $((1 << 27))
```

performs similarly to going over the NIC within a node:

```bash
mem_type: pinned_host
p2p_size: 134217728
[-1] time: 0.0142898
Doing MPI_Send/Recv from rank 0 to rank 1, of 134217728 bytes
[0:0] time: 0.00710427
[0:1] time: 0.0056088
[0:2] time: 0.00560714
[0:3] time: 0.00560653
[0:4] time: 0.00560583
[1:0] time: 0.00701528
[1:1] time: 0.00560708
[1:2] time: 0.00560733
[1:3] time: 0.00560672
[1:4] time: 0.00560602
```

Running the program without CPU binding:

```bash
srun -n 2 -N 1 --cpu-bind=none internode_pinned_host_comm 2 5 host $((1 << 27))
```

performs better than the bound case, but not as well as the NIC case. CPU
bindings are typically required to otherwise get good performance:

```bash
mem_type: pinned_host
p2p_size: 134217728
[-1] time: 3.91474
Doing MPI_Send/Recv from rank 0 to rank 1, of 134217728 bytes
[0:0] time: 0.00950821
[0:1] time: 0.0101551
[0:2] time: 0.00936332
[0:3] time: 0.0092744
[0:4] time: 0.00961944
[1:0] time: 0.0112646
[1:1] time: 0.0100813
[1:2] time: 0.00953244
[1:3] time: 0.00978142
[1:4] time: 0.0095964
```

Communicating from unpinned host memory, with or without the NIC:

```bash
srun -n 2 -N 1 --cpu-bind=sockets internode_pinned_host_comm 2 5 host $((1 << 27))
```

performs better than all the cases above, but similarly to when using the NIC
(approximately 2x faster than the NIC case, approximately 8x faster than the
CMA pinned host memory case):

```bash
mem_type: host
p2p_size: 134217728
[-1] time: 3.14696
Doing MPI_Send/Recv from rank 0 to rank 1, of 134217728 bytes
[0:0] time: 0.00828594
[0:1] time: 0.00282664
[0:2] time: 0.00293067
[0:3] time: 0.002938
[0:4] time: 0.00282891
[1:0] time: 0.00903648
[1:1] time: 0.00297793
[1:2] time: 0.00294309
[1:3] time: 0.00290744
[1:4] time: 0.00287154
```

Communicating smaller does not seem to suffer as much when using pinned host memory.
