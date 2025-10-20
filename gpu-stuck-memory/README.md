# OS-allocated file caches in GPU memory prevent GPU memory allocation

Due to the unified memory architecture of GH200, linux will sometimes use GPU memory for file caches.
When an application allocates GPU memory directly, the GPU driver should evict the file caches to CPU memory.
However, a bug in versions of the driver prior to 570 did not do this properly, leading to out of memory errors for applications that are IO heavy. 

## Prerequisites

The test requires [fio](https://github.com/axboe/fio) to be installed and available in `PATH`.

# Reproducer

This simple reproducer does three things:
- Allocate 95% of GPU memory. This should pass on a freshly assigned node. 
- Create and read a 100GB file on scratch.
- Attempt to allocate 95% of GPU memory again.

This test should pass if the driver correctly evicts OS file caches, but fails on the second allocation attempt if the bug is present.

# Assumptions

This test assumes that the node being tested has already had the file caches flushed before the node was allocated to the job.
This is currently (20.10.2025) the case during one of the slurm prolog scripts.

