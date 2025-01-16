# Particular block sizes cause task to hang when writing to lustre file system

Jobs run on the ML Platform vCluster Clariden were observed to hang when the
slurm job time limit expired or the job was cancelled.  Slurm would report the
nodes stuck in the COMPLETING state.  

A python tokenizing script was observed to consistently cause this problem.
Further investigation found that writes from Python are buffered. The
buffer size is a multiple of what python thinks is the block size of the file
system and appears to grow in response to the size of data written at once.

As of 16-Jan-2025 on capstor and iopsstor this value is

```bash
$ touch tmpfile.txt
$ python3 -c "import os; print(os.stat('tmpfile.txt').st_blksize)"
4194304
```

However, `lfs` and `stat` report different values

```bash
$ lfs getstripe tmpfile.txt
tmpfile.txt
lmm_stripe_count:  1
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    0
lmm_stripe_offset: 18
	obdidx		 objid		 objid		 group
	    18	      64606835	    0x3d9d273	             0

$ stat -f tmpfile.txt
  File: "tmpfile.txt"
    ID: 517b58b800000000 Namelen: 255     Type: lustre
Block size: 4096       Fundamental block size: 4096
Blocks: Total: 808226981590 Free: 299207122227 Available: 291055212694
Inodes: Total: 3193167946 Free: 3074713722
```

The `python-io-demo.py` code writes different array sizes to a file. Using `strace` one can see how the data is buffered and actually written to disk.
This demo uses both buffered and unbuffered mode.

```bash
strace python3 python-io-demo.py 2>&1 | grep -e write -e Writing
```

## The Reproducer

To reproduce the issue, the submission script runs `dd` with a block size of
4096000, which is not exactly the buffer size python uses and not a
power-of-two multiple of the lustre block size. The script also tests other
simple block sizes which are a power-of-two. Only the odd size has been
observed to cause at least one of the tasks to hang. Using more than one task
is more likely to generate a problem, although the issue has been observed with
a single task.

Once the process has hung the job will timeout (or if cancelled via scancel) and
slurm will remain in the COMPLETING state.

## Run

The following submission script is configured to use 1 node with 2 tasks.

```bash
sbatch sbatch_dd_repro.sh
```
