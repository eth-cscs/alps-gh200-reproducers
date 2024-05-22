# Slow first communication from GPU buffers whose size are not multiples of the page size

The first communication from a large GPU buffer, even when communicating only a
fraction of the full buffer, is very slow if the large GPU buffer is not page-
or large-page-aligned. Subsequent communication is fast. Only the sender side
seems to be affected.

## Build

With an environment which has `g++`, CUDA, and GPU-aware MPICH loaded:

```bash
g++ -O3 -lcudart -lmpi -lmpi_gtl_cuda gpudirect_p2p_overalloc.cpp -o gpudirect_p2p_overalloc
```

## Run

```bash
srun -n 2 gpudirect_p2p_overalloc <num_iterations> <num_sub_iterations> <p2p_size> <buffer_size> <overalloc_size>
```

The test will do an `MPI_Send/Recv` from the first to the last rank of
`<p2p_size>` bytes from a GPU buffer of `<buffer_size>` and `<buffer_size> +
<overalloc_size>` bytes. "Overallocation" refers to allocating slightly more
memory than is actually needed. The test will perform one warmup iteration with
a small allocation, and then `<num_iterations>` each with and without
overallocation. Each iteration performs `<num_sub_iterations>` `MPI_Send/Recv`s
with the same allocation.

Note that `<p2p_size>` must be at least `MPICH_GPU_IPC_THRESHOLD` (which
defaults to 1KiB) to see the effect below. Otherwise communication is done via
CPU memory.

To communicate 1KiB from a buffer of 1GiB and a 1B overallocation:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 3 $((1 << 10)) $((1 << 30)) 1
```

The program will likely output something like:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 1
[-1] time: 2.17581
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741825 bytes (with overalloc).
[0:0] time: 0.454265
[0:1] time: 2.8895e-05
[0:2] time: 1.9328e-05
[1:0] time: 0.456158
[1:1] time: 0.00184436
[1:2] time: 3.4303e-05
[2:0] time: 0.454067
[2:1] time: 2.6175e-05
[2:2] time: 1.8111e-05
[3:0] time: 0.458347
[3:1] time: 2.3487e-05
[3:2] time: 2.1439e-05
[4:0] time: 0.462559
[4:1] time: 2.4671e-05
[4:2] time: 1.6959e-05
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0:0] time: 0.000880008
[0:1] time: 2.192e-05
[0:2] time: 1.8527e-05
[1:0] time: 0.00345239
[1:1] time: 2.1087e-05
[1:2] time: 1.9071e-05
[2:0] time: 0.00360752
[2:1] time: 2.144e-05
[2:2] time: 1.8271e-05
[3:0] time: 0.00340928
[3:1] time: 2.4223e-05
[3:2] time: 1.8272e-05
[4:0] time: 0.00341379
[4:1] time: 2.2207e-05
[4:2] time: 1.808e-05
```

The overallocated case is two orders of magnitude slower.

Overallocating a 64KiB page size:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 3 $((1 << 10)) $((1 << 30)) $((1 << 16))
```

will likely result in something like:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 65536
[-1] time: 2.14635
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073807360 bytes (with overalloc).
[0:0] time: 0.0438379
[0:1] time: 2.3072e-05
[0:2] time: 1.7408e-05
[1:0] time: 0.0483795
[1:1] time: 3.1871e-05
[1:2] time: 1.8783e-05
[2:0] time: 0.047749
[2:1] time: 2.2303e-05
[2:2] time: 1.6672e-05
[3:0] time: 0.0484019
[3:1] time: 2.1024e-05
[3:2] time: 2.1247e-05
[4:0] time: 0.0483549
[4:1] time: 2.2175e-05
[4:2] time: 1.792e-05
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0:0] time: 0.000901063
[0:1] time: 2.0415e-05
[0:2] time: 1.7696e-05
[1:0] time: 0.00336631
[1:1] time: 2.1536e-05
[1:2] time: 1.7888e-05
[2:0] time: 0.0033328
[2:1] time: 2.0351e-05
[2:2] time: 1.7344e-05
[3:0] time: 0.00331002
[3:1] time: 2.4992e-05
[3:2] time: 1.9168e-05
[4:0] time: 0.00333485
[4:1] time: 2.1695e-05
[4:2] time: 1.7855e-06
```

The overallocated case is "only" one order of magnitude slower.

Overallocating a large 2MiB page size:

```bash
srun -u -n 2 $PWD/gpudirect_p2p_overalloc 5 3 $((1 << 10)) $((1 << 30)) $((1 << 21))
```

results in the same performance in both cases:

```bash
p2p_size: 1024
buffer_size: 1073741824
overalloc_size: 2097152
[-1] time: 2.14697
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1075838976 bytes (with overalloc).
[0:0] time: 0.000898375
[0:1] time: 2.2816e-05
[0:2] time: 1.7471e-05
[1:0] time: 0.0026861
[1:1] time: 3.0431e-05
[1:2] time: 1.8367e-05
[2:0] time: 0.0026094
[2:1] time: 2.1183e-05
[2:2] time: 1.8336e-05
[3:0] time: 0.00254108
[3:1] time: 2.1216e-05
[3:2] time: 2.2048e-05
[4:0] time: 0.0025654
[4:1] time: 2.0704e-05
[4:2] time: 1.824e-05
Doing MPI_Send/Recv from rank 0 to rank 1, of 1024 bytes from a buffer of 1073741824 bytes (without overalloc).
[0:0] time: 0.0008802
[0:1] time: 1.9359e-05
[0:2] time: 1.696e-05
[1:0] time: 0.00345136
[1:1] time: 2.0895e-05
[1:2] time: 1.792e-05
[2:0] time: 0.00341987
[2:1] time: 1.9744e-05
[2:2] time: 1.712e-05
[3:0] time: 0.00336791
[3:1] time: 2.5631e-05
[3:2] time: 1.7856e-05
[4:0] time: 0.00331661
[4:1] time: 2.0672e-05
[4:2] time: 1.7824e-05
```

Communicating the whole `<buffer_size>` is also slower with overallocation:

```bash
srun -u -n 2 gpudirect_p2p_overalloc 5 3 $((1 << 30)) $((1 << 30)) 1
```

```bash
p2p_size: 1073741824
buffer_size: 1073741824
overalloc_size: 1
[-1] time: 2.12013
Doing MPI_Send/Recv from rank 0 to rank 1, of 1073741824 bytes from a buffer of 1073741825 bytes (with overalloc).
[0:0] time: 0.46222
[0:1] time: 0.00808007
[0:2] time: 0.00807456
[1:0] time: 0.469106
[1:1] time: 0.00809424
[1:2] time: 0.00807559
[2:0] time: 0.468073
[2:1] time: 0.00808419
[2:2] time: 0.00807872
[3:0] time: 0.466539
[3:1] time: 0.00808035
[3:2] time: 0.00808195
[4:0] time: 0.469327
[4:1] time: 0.00808474
[4:2] time: 0.00807716
Doing MPI_Send/Recv from rank 0 to rank 1, of 1073741824 bytes from a buffer of 1073741824 bytes (without overalloc).
[0:0] time: 0.0116373
[0:1] time: 0.00808311
[0:2] time: 0.00808387
[1:0] time: 0.0115
[1:1] time: 0.00807831
[1:2] time: 0.00807773
[2:0] time: 0.0115344
[2:1] time: 0.00807677
[2:2] time: 0.00807847
[3:0] time: 0.0114708
[3:1] time: 0.00808624
[3:2] time: 0.00807773
[4:0] time: 0.0115159
[4:1] time: 0.00808125
[4:2] time: 0.00807597
```
