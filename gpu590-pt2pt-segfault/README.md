# GPU-aware MPI causes segfault

OSU bandwidth (point-to-point) causes a segfault on starlex. This currently has SLES15.6 and Nvidia driver 590.
Alltoall is ok.

## Prerequisites

The test requires [fio](https://github.com/axboe/fio) to be installed and available in `PATH`.

The reproducer depends on three uenvs (these are pulled automatically in the script):

```
prgenv-gnu/25.11:v1
prgenv-gnu-openmpi/25.12:v1@daint
service::prgenv-gnu/25.11:test1@starlex
```

# Reproducer

The main reproducer (`gpu590-bug-uenv.sh`) allocates two tasks, first on the same node and then on separate nodes.

It the executes
```
srun osu_alltoall H H
srun osu_alltoall D D
srun osu_bw H H
srun osu_bw H D
srun osu_bw D H
srun osu_bw D D
```

