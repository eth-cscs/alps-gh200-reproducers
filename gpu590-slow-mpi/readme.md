# GPU-590 slow MPI

CPU-only MPI communication slowed down significantly with the new SLES 15sp6 + driver update.

This was observed in CP2K, in part of the code that performs Scalapack routines on the CPU.

The `pdsygst` example was extracted to demonstrate this. It performs three stages:

* `pdgemr2d`: distribute initial matrix state from root rank
- `pdpotrf`: perform distributed cholesky solve
- `pdsygst` perform distributed eigen solve

## Key facts

- the upgraded image upgrades the NVIDIA driver from 555 to 590
- the upgraded image upgrades the Linux kernel from 5.14.21 to 6.4.0
- the problem can be observed when running on a single node: it is an intra-node shared memory issue.
- the tests here are "CPU only"
    - the `pdsygst` benchmark was extracted from a GPU application, but represents part of the code that runs entirely on the CPU
- we are using exactly the same executable with the same uenv mounted on both systems
    - this has been observed using old uenv that use libfabric 1.15.2 from the system and the latest uenv that package open source libfabric, CXI, etc.

## Notes

### running the benchmark

Building the example should be performed on Daint

- build a single executable on Daint and use for tests on both Daint and Starlex
- use the same uenv on both systems
- executables built on Starlex link against symbols from new glibc, and can't be run on Daint.

```
uenv start --view=default prgenv-gnu/25.6:v2@daint
mkdir build; cd build
CC=gcc CXX=g++ cmake ..
make
```

The benchmark is not compute bound, and because the matrix size is relatively small we suspect that OpenBLAS falls back to single threaded implementations.
So we use a single core per rank:

- reproduces the behavior with multiple cores
- simplifies reproducing the issue
- reduces the likelihood that thread-core affinity could be a root cause.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun -n32 -c1 -N1 -Acsstaff ./pdsygst
```

### Timings with `MPICH_GPU_SUPPORT_ENABLED=0`

Results on both systems are very close when the benchmark is run with GPU aware MPI explicitly disabled

```
Daint:

pdgemr2d:  0.145,  [ 0.103,  0.185] seconds
pdpotrf :  0.218,  [ 0.218,  0.218] seconds
pdsygst :  1.489,  [ 1.488,  1.490] seconds

Starlex:

pdgemr2d:  0.142,  [ 0.101,  0.182] seconds
pdpotrf :  0.218,  [ 0.218,  0.218] seconds
pdsygst :  1.435,  [ 1.434,  1.436] seconds
```

### Timings with `MPICH_GPU_SUPPORT_ENABLED=1`

- results are much slower than without GPU support on both systems
- matrix distribution via `pdgemr2d` is not affected
- the `pdpotrf` (cholesky) and `pdsygst` (eigen) solvers are slower
- the results on Daint and Starlex are significantly different: Starlex is 2x slower at `pdsygst`

```
Daint:

pdgemr2d:  0.140,  [ 0.101,  0.178] seconds
pdpotrf :  4.339,  [ 4.339,  4.339] seconds
pdsygst :  2.287,  [ 2.286,  2.288] seconds

Starlex:

pdgemr2d:  0.144,  [ 0.104,  0.188] seconds
pdpotrf :  3.134,  [ 3.134,  3.134] seconds
pdsygst :  5.762,  [ 5.760,  5.764] seconds
```

There are two problems here:

1. the universal problem that GPU-aware MPI has an enormous negative impact on performance.
2. the local problem that this got significantly worse with the system upgrade.

### use strace to check for system call differences with/without `MPICH_GPU_SUPPORT_ENABLED`

Use the following wrapper to collect strace statistics on rank 0:

```
#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ $SLURM_PROCID -eq 0 ]; then
    strace -c -o strace.$CLUSTER_NAME.out ./pdsygst
    cat strace.$CLUSTER_NAME.out
else
    ./pdsygst
fi
```

results with `MPICH_GPU_SUPPORT_ENABLED=0`:

```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- -----------------------
 60.46    0.604353        2674       226           lseek
 31.73    0.317157        1201       264           read
  5.45    0.054432           0    118012           sched_yield
  0.95    0.009457          31       299       254 openat
  0.75    0.007487          23       319           munmap
  0.28    0.002824           7       388           mmap
  0.11    0.001120           8       128           ioctl
  ...
------ ----------- ----------- --------- --------- -----------------------
100.00    0.999583           8    120190       261 total
```

**STARLEX** results with `MPICH_GPU_SUPPORT_ENABLED=1`:

```
pdgemr2d:  0.159,  [ 0.115,  0.201] seconds
pdpotrf :  3.134,  [ 3.134,  3.134] seconds
pdsygst :  5.912,  [ 5.910,  5.913] seconds
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- -----------------------
 52.08    2.192327        2262       969           ioctl
 23.83    1.003070           1    923316           sched_yield
 14.68    0.617995        2722       227           lseek
  7.86    0.330830         705       469           read
  0.89    0.037260           4      8490           process_vm_readv
  0.28    0.011967           1      9629           getpid
  0.17    0.007006          21       323           munmap
  0.07    0.002804           6       463       262 openat
------ ----------- ----------- --------- --------- -----------------------
100.00    4.209411           4    945351       316 total
```

**DAINT**  results with `MPICH_GPU_SUPPORT_ENABLED=1`:

```
pdgemr2d:  0.147,  [ 0.108,  0.186] seconds
pdpotrf :  4.281,  [ 4.281,  4.281] seconds
pdsygst :  2.412,  [ 2.411,  2.413] seconds
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- -----------------------
 59.82    3.341055        3483       959           ioctl
 15.79    0.881926           0    884327           sched_yield
 13.13    0.733562        3161       232           lseek
  6.20    0.346349         671       516           read
  2.33    0.130137         400       325           munmap
  1.49    0.083150          70      1173       944 openat
  0.45    0.025193           2      8490           process_vm_readv
  0.20    0.011045          53       207       127 newfstatat
  0.19    0.010687          19       548           mmap
  0.14    0.007878           0      9639           getpid
  0.10    0.005676         946         6           getdents64
------ ----------- ----------- --------- --------- -----------------------
100.00    5.585459           6    907501      1116 total
```

From these results:

- `ioctl` and `sched_yield` dominate with GPU-aware, while accounting for 5% of total system overhead with GPU-aware MPI disabled
- The system call overhead on both Daint and Starlex is high (5.58s and 4.21s respectively) compared to the system time of 1s in GPU-aware mode
- Daint spends more time in system calls than Starlex
    - the total slower runtime difference is not due to that overhead alone
    - I am running with a single core on each rank, so there should be some level of serialization.

### Using linaro map:

```
uenv start --view=prgenv-gnu:default prgenv-gnu/25.6:v2@daint,linaro-forge/25.1@daint
export PATH=$PATH:/user-tools/env/default/bin
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun -n32 -c1 -N1 -Acsstaff ./pdsygst
#pdgemr2d:  0.172,  [ 0.133,  0.210] seconds
#pdpotrf :  3.145,  [ 3.145,  3.146] seconds
#pdsygst :  5.729,  [ 5.727,  5.730] seconds
map -n 32 --mpi=slurm --mpiargs="-c1 -N1" --profile ./pdsygst
#pdgemr2d:  0.187,  [ 0.140,  0.229] seconds
#pdpotrf :  2.186,  [ 2.186,  2.186] seconds
#pdsygst : 22.399,  [22.396, 22.401] seconds
#
#MAP analysing program...
#MAP gathering samples...
#MAP generated /users/bcumming/software/fortran-playground/pdsygst/build/pdsygst_32p_1n_1t_2026-02-27_10-25.map
#
#Linaro MAP profiling summary
#============================
#Profiling time:      26 seconds (between MPI init and finalize)
#Peak process memory: 1606418432 B (~1.50 GiB)
#
#Compute:               2.1%     (0.5s) ||
#MPI:                  97.9%    (25.2s) |=========|
#I/O:                   0.0%     (0.0s) |
```

Findings:
- time taken in pdsygst explodes when run in MAP
- the other parts are faster/the same

