# Slow first communication from GPU buffers whose size are not multiples of the page size

The first communication from a large GPU buffer, even when communicating only a
fraction of the full buffer, is very slow if the large GPU buffer is not page-
or large-page-aligned. Subsequent communication is fast. Only the sender side
seems to be affected.

## Build

With an environment which has `g++`, CUDA, and GPU-aware MPICH loaded:

```bash
g++ -lcudart -lmpi -lmpi_gtl_cuda gpudirect_p2p_overalloc.cpp -o gpudirect_p2p_overalloc
```

## Run

```bash
srun -n 2 gpudirect_p2p_overalloc <num_iterations> <p2p_size> <buffer_size> <overalloc_size>
```

The test will do an `MPI_Send/Recv` from the first to the last rank of
`<p2p_size>` bytes from a GPU buffer of `<buffer_size>` and `<buffer_size> +
<overalloc_size>` bytes. "Overallocation" refers to allocating slightly more
memory than is actually needed. The test will perform one warmup iteration with
a small allocation, and then `<num_iterations>` each with and without
overallocation. Each iteration does a new allocation.

Note that `<p2p_size>` must be at least `MPICH_GPU_IPC_THRESHOLD` (which
defaults to 1KiB) to see the effect below. Otherwise communication is done via
CPU memory.

To communicate 1KiB from a buffer of 1GiB and a 1B overallocation:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 $((1 << 10)) $((1 << 30)) 1
```

The program will likely output something like:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 1
[-1] time: 3.96283
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741825 bytes (with overalloc).
[0] time: 0.461843
[1] time: 0.463472
[2] time: 0.464589
[3] time: 0.462855
[4] time: 0.464711
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0] time: 0.00383505
[1] time: 0.00516193
[2] time: 0.00510055
[3] time: 0.00514718
[4] time: 0.00515585
```

The overallocated case is two orders of magnitude slower.

Overallocating a 64KiB page size:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 $((1 << 10)) $((1 << 30)) $((1 << 16))
```

will likely result in something like:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 65536
[-1] time: 3.98771
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073807360 bytes (with overalloc).
[0] time: 0.0475203
[1] time: 0.0508146
[2] time: 0.0498883
[3] time: 0.0504232
[4] time: 0.0523863
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0] time: 0.00379333
[1] time: 0.00576192
[2] time: 0.00517134
[3] time: 0.00516142
[4] time: 0.00509916
```

The overallocated case is "only" one order of magnitude slower.

Overallocating a large 2MiB page size:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 $((1 << 10)) $((1 << 30)) $((1 << 21))
```

results in the same performance in both cases:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 2097152
[-1] time: 4.01528
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1075838976 bytes (with overalloc).
[0] time: 0.0047899
[1] time: 0.00574066
[2] time: 0.00623194
[3] time: 0.00604808
[4] time: 0.0067584
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0] time: 0.00446229
[1] time: 0.00618366
[2] time: 0.00652358
[3] time: 0.00646784
[4] time: 0.0065071
```

Communicating the whole `<buffer_size>` is also slower with overallocation:

```bash
srun -u -n 2 gpudirect_p2p_overalloc 5 $((1 << 30)) $((1 << 30)) 1
```

```bash
p2p_size: 1073741824
buffer_size: 1073741824
overalloc_size: 1
[-1] time: 3.93228
Doing MPI_Send/Recv from rank 0 to rank 1, of 1073741824 bytes from a buffer of 1073741825 bytes (with overalloc).
[0] time: 0.467979
[1] time: 0.475299
[2] time: 0.470099
[3] time: 0.474272
[4] time: 0.468992
Doing MPI_Send/Recv from rank 0 to rank 1, of 1073741824 bytes from a buffer of 1073741824 bytes (without overalloc).
[0] time: 0.0161709
[1] time: 0.0184779
[2] time: 0.0160849
[3] time: 0.0163321
[4] time: 0.0160386
```
