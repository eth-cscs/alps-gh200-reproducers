#!/bin/bash

function fail()
{
    echo $1
    exit 1
}

FASTDISK=/iopsstor/scratch/cscs/jpcoles/
LARGEFILE=${FASTDISK}/gpu-stuckmem.tmp

nvidia-smi
/usr/bin/parallel_allocate_free_gpu_mem 95 || fail "First allocation failed."

fio --name=cachetest --rw=read --size=100G --filename=${LARGEFILE} --bs=1M --ioengine=sync --direct=0

nvidia-smi
/usr/bin/parallel_allocate_free_gpu_mem 95 || fail "Last allocation failed."
