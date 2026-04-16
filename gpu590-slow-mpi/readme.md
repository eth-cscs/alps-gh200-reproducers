# GPU-590 slow MPI

**HPE Case** a ticket has been opened with HPE: 5402335846

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

### Running the benchmark

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

# Nsys

> This profiling is performed _after_ the update of Daint, where Daint was aligned with Starlex.
> Therefore, the comparison is betweenm `MPICH_GPU_SUPPORT_ENABLED=0` and `MPICH_GPU_SUPPORT_ENABLED=1`
> with the new driver, rather than between different drivers.

The issue can be reproduced also with two ranks on a single node:

```
$ MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 srun -A csstaff -p debug -N 1 --ntasks-per-node 2 --cpus-per-task 1 build/pdsygst
pdgemr2d:  0.225,  [ 0.225,  0.225] seconds
pdpotrf :  1.880,  [ 1.880,  1.880] seconds
pdsygst :  6.473,  [ 6.473,  6.473] seconds
$ MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 srun -A csstaff -p debug -N 1 --ntasks-per-node 2 --cpus-per-task 1 build/pdsygst
pdgemr2d:  0.226,  [ 0.226,  0.227] seconds
pdpotrf :  3.500,  [ 3.500,  3.500] seconds
pdsygst : 13.789,  [13.789, 13.789] seconds
```

Profiling with `--trace=mpi` does not work; it results in a segmentation fault. Profiling with `--trace=osrt` instead.

**`MPICH_GPU_SUPPORT_ENABLED=0`**:

```
 ** OS Runtime Summary (osrt_sum):                                                                                                                                     
 Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)          Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ---------------------
     21.8          618,016        294   2,102.1   1,824.0     1,312    20,416      1,653.5  munmap
     18.8          534,592          9  59,399.1  15,712.0     3,360   340,576    108,152.4  open
     16.3          462,912         40  11,572.8  12,416.0     1,920    16,544      3,331.7  stat
      9.2          262,624          9  29,180.4  30,240.0     2,144    57,632     20,658.1  ioctl
      8.8          249,856          5  49,971.2  35,808.0    20,384   124,352     42,180.0  fopen
      6.5          185,920          3  61,973.3  63,200.0    42,944    79,776     18,446.6  write
      6.5          183,776          3  61,258.7  62,464.0    58,848    62,464      2,087.7  usleep
      3.2           90,432          7  12,918.9  12,640.0     5,152    22,336      6,737.2  mmap
      1.5           43,680          4  10,920.0  11,360.0     5,280    15,680      5,543.1  shmat
      1.5           43,296          4  10,824.0  10,048.0     6,560    16,640      5,012.1  shmdt
      1.4           38,656          4   9,664.0   4,832.0     3,872    25,120     10,333.0  shmget
      1.2           34,400          5   6,880.0   1,312.0     1,088    28,608     12,155.2  close
      0.9           25,408          6   4,234.7   3,168.0     1,248     9,984      3,398.6  ftruncate
      0.7           18,592          2   9,296.0   9,296.0     9,280     9,312         22.6  fgets
      0.4           12,416          4   3,104.0   1,376.0     1,056     8,608      3,672.8  fflush
      0.4           11,552          3   3,850.7   4,192.0     2,464     4,896      1,251.4  fclose
      0.2            7,072          1   7,072.0   7,072.0     7,072     7,072          0.0  fread
      0.2            6,304          2   3,152.0   3,152.0     2,816     3,488        475.2  fstat
      0.2            4,576          3   1,525.3   1,184.0     1,120     2,272        647.4  shmctl
      0.1            3,200          1   3,200.0   3,200.0     3,200     3,200          0.0  alarm
      0.1            2,368          2   1,184.0   1,184.0     1,120     1,248         90.5  pthread_mutex_trylock
      0.0            1,056          1   1,056.0   1,056.0     1,056     1,056          0.0  signal   
```
According to `nsys`, `libc` accounts for 1% of the total runtime.


**`MPICH_GPU_SUPPORT_ENABLED=1`**:
```
 ** OS Runtime Summary (osrt_sum):                                                                                                                                     
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)            Name
 --------  ---------------  ---------  ------------  -------------  ---------  -----------  ------------  ----------------------
     97.3   49,516,578,400        498  99,430,880.3  100,129,680.0      1,696  939,167,776  40,445,432.6  poll
      2.1    1,057,431,520        911   1,160,737.1       16,288.0      1,056  462,323,520  19,003,143.1  ioctl
      0.5      276,857,376     16,447      16,833.3        7,200.0      3,488   16,138,112     206,436.2  process_vm_readv
      0.0        9,057,088          1   9,057,088.0    9,057,088.0  9,057,088    9,057,088           0.0  pthread_cond_wait
      0.0        1,554,464          3     518,154.7      507,296.0    482,944      564,224      41,713.8  pthread_create
      0.0          914,208         16      57,138.0       29,936.0      1,088      318,656      80,520.2  write
      0.0          899,968         56      16,070.9       15,792.0      3,904       37,216       5,807.4  mmap64
      0.0          832,992         75      11,106.6        6,592.0      1,568       49,088      10,581.9  fopen
      0.0          806,208         82       9,831.8        9,216.0      3,072       18,592       3,241.9  open64
      0.0          771,456        297       2,597.5        1,952.0      1,376       23,200       2,430.4  munmap
      0.0          682,016         44      15,500.4       15,136.0      1,728       40,448       7,674.7  mmap
      0.0          454,816          1     454,816.0      454,816.0    454,816      454,816           0.0  pthread_cond_broadcast
      0.0          275,136          1     275,136.0      275,136.0    275,136      275,136           0.0  sem_timedwait
      0.0          247,520         17      14,560.0       10,496.0      2,144       54,496      14,072.5  open
      0.0          201,760          4      50,440.0       33,936.0     18,240      115,648      45,006.5  shmdt
      0.0          193,120          3      64,373.3       62,944.0     57,344       72,832       7,842.3  usleep
      0.0          131,104         52       2,521.2        1,568.0      1,024       15,296       2,610.9  fclose
      0.0           95,040         38       2,501.1        2,272.0      1,408        8,064       1,156.3  stat
      0.0           71,904         13       5,531.1        3,296.0      1,056       16,992       4,963.9  close
      0.0           71,264          3      23,754.7       11,936.0      9,088       50,240      22,981.1  fgets
      0.0           48,064          5       9,612.8        6,240.0      4,608       16,448       5,743.7  shmat
      0.0           43,392          5       8,678.4        5,760.0      3,776       21,760       7,378.4  shmget
      0.0           30,784          4       7,696.0        7,888.0      2,656       12,352       4,249.0  pipe2
      0.0           18,560          5       3,712.0        2,752.0      1,568        6,720       2,117.2  ftruncate
      0.0           17,824          9       1,980.4        1,792.0      1,312        3,456         789.9  read
      0.0           15,360          2       7,680.0        7,680.0      7,648        7,712          45.3  socket
      0.0           12,000          2       6,000.0        6,000.0      1,280       10,720       6,675.1  fflush
      0.0           11,680          1      11,680.0       11,680.0     11,680       11,680           0.0  connect
      0.0            8,512          5       1,702.4        1,792.0      1,344        2,080         329.0  shmctl
      0.0            5,280          2       2,640.0        2,640.0      1,984        3,296         927.7  fwrite
      0.0            3,648          1       3,648.0        3,648.0      3,648        3,648           0.0  fread
      0.0            3,584          2       1,792.0        1,792.0      1,056        2,528       1,040.9  alarm
      0.0            2,880          2       1,440.0        1,440.0      1,408        1,472          45.3  fstat
      0.0            2,848          1       2,848.0        2,848.0      2,848        2,848           0.0  bind
      0.0            1,152          1       1,152.0        1,152.0      1,152        1,152           0.0  signal
      0.0            1,120          1       1,120.0        1,120.0      1,120        1,120           0.0  listen  
```
According to `nsys`, `libc` accounts for 18% of the total runtime.
