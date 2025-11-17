# Updates

- 17-Nov-2025
    - HPE identified a problem caused when the page size of the client and server differ
    - On arm the page size is 64k, while on the x86 server it is 4k.
    - A patch and subsequent testing has finally shown the bug to be fixed.
    - dd_repro-scan-large-fast.sh was the final script used for testing on starlex.

- 18-Jan-2025
    - The hanging behavior is clearly associated with the LNetError that appears in the output of dmesg as it shows up precisely when the process hangs.
    - The error won't *always* appear, even for the same block size, but after at most a few attempts it will hang. Two tasks are more likely to reproduce the problem, but one task has been enough.
    - The precise block size isn't the critical factor, as I have found a number of other (smaller) block sizes that also cause a hang.
    - This seems to be very specific to capstor. I haven't been able to reproduce the problem on iopsstor.

# Particular block sizes cause task to hang when writing to lustre file system

Jobs run on the ML Platform vCluster Clariden were observed to hang when the
slurm job time limit expired or the job was cancelled.  Slurm would report the
nodes stuck in the COMPLETING state.  

A Python ML tokenizing script was observed to consistently cause this problem,
although the issue has been observed from codes written in, e.g., C++.

Further investigation using the Python reproducer found that writes from Python
are buffered. The buffer size is a multiple of what Python thinks is the block
size of the file system and appears to grow in response to the size of data
written at once.

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

The case that fails was when 100\*1024\*8=819200 bytes were written
continuously. Python packs 5 of these (4096000 bytes) into one buffer
(presumably because 6 would be too big) before issuing a single system call to
write the buffer.

## The Reproducer

To reproduce the issue, the submission script runs `dd` with a block size of
4096000, which is not exactly the full buffer size Python uses and not a
power-of-two multiple of the lustre block size. The script also tests other
simple block sizes which are a power-of-two. Only the odd size has been
observed to cause at least one of the tasks to hang. Using more than one task
is more likely to generate a problem, although the issue has been observed with
a single task.

Once the process has hung the job will timeout (or if cancelled via scancel) and
slurm will remain in the COMPLETING state.

The following messages from `dmesg -T` started at the time of the hang endlessly repeat:

```
[Wed Jan 15 17:19:05 2025] LNetError: 4069:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.43@tcp, match 1820705849269056 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:22:28 2025] Lustre: 128727:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736957925/real 1736957925]  req@000000005ab3b574 x1820705849269056/t0(0) o4->capstor-OST003a-osc-ffff3000ef06c000@172.28.1.43@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958128 ref>
[Wed Jan 15 17:22:28 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection to capstor-OST003a (at 172.28.1.43@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:22:28 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.43@tcp)
[Wed Jan 15 17:22:28 2025] LNetError: 4108:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.43@tcp, match 1820705849565760 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:22:28 2025] LNetError: 4108:0:(lib-ptl.c:189:lnet_try_match_md()) Skipped 1 previous similar message
[Wed Jan 15 17:22:30 2025] Lustre: 128781:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736957926/real 1736957926]  req@00000000ceb3e0bd x1820705849272448/t0(0) o4->capstor-OST005a-osc-ffff3000ef06c000@172.28.1.59@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958131 ref>
[Wed Jan 15 17:22:30 2025] Lustre: capstor-OST005a-osc-ffff3000ef06c000: Connection to capstor-OST005a (at 172.28.1.59@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:22:30 2025] Lustre: capstor-OST005a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.59@tcp)
[Wed Jan 15 17:22:30 2025] LNetError: 4091:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.59@tcp, match 1820705849572736 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:25:51 2025] Lustre: 128727:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736958128/real 1736958128]  req@000000005ab3b574 x1820705849269056/t0(0) o4->capstor-OST003a-osc-ffff3000ef06c000@172.28.1.43@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958331 ref>
[Wed Jan 15 17:25:51 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection to capstor-OST003a (at 172.28.1.43@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:25:51 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.43@tcp)
[Wed Jan 15 17:25:51 2025] LNetError: 4099:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.43@tcp, match 1820705849854848 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:25:55 2025] Lustre: 128781:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736958131/real 1736958131]  req@00000000ceb3e0bd x1820705849272448/t0(0) o4->capstor-OST005a-osc-ffff3000ef06c000@172.28.1.59@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958336 ref>
[Wed Jan 15 17:25:55 2025] Lustre: capstor-OST005a-osc-ffff3000ef06c000: Connection to capstor-OST005a (at 172.28.1.59@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:25:55 2025] Lustre: capstor-OST005a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.59@tcp)
[Wed Jan 15 17:25:55 2025] LNetError: 4071:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.59@tcp, match 1820705849857152 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:29:13 2025] Lustre: 128727:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736958331/real 1736958331]  req@000000005ab3b574 x1820705849269056/t0(0) o4->capstor-OST003a-osc-ffff3000ef06c000@172.28.1.43@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958534 ref>
[Wed Jan 15 17:29:14 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection to capstor-OST003a (at 172.28.1.43@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:29:14 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.43@tcp)
[Wed Jan 15 17:29:14 2025] LNetError: 4107:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.43@tcp, match 1820705850136256 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:32:36 2025] Lustre: 128727:0:(client.c:2310:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: [sent 1736958534/real 1736958534]  req@000000005ab3b574 x1820705849269056/t0(0) o4->capstor-OST003a-osc-ffff3000ef06c000@172.28.1.43@tcp:6/4 lens 488/448 e 2 to 1 dl 1736958737 ref>
[Wed Jan 15 17:32:36 2025] Lustre: 128727:0:(client.c:2310:ptlrpc_expire_one_request()) Skipped 1 previous similar message
[Wed Jan 15 17:32:36 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection to capstor-OST003a (at 172.28.1.43@tcp) was lost; in progress operations using this service will wait for recovery to complete
[Wed Jan 15 17:32:36 2025] Lustre: Skipped 1 previous similar message
[Wed Jan 15 17:32:36 2025] Lustre: capstor-OST003a-osc-ffff3000ef06c000: Connection restored to  (at 172.28.1.43@tcp)
[Wed Jan 15 17:32:36 2025] Lustre: Skipped 1 previous similar message
[Wed Jan 15 17:32:36 2025] LNetError: 4070:0:(lib-ptl.c:189:lnet_try_match_md()) Matching packet from 12345-172.28.1.43@tcp, match 1820705850437120 length 1048576 too big: 1015808 left, 1015808 allowed
[Wed Jan 15 17:32:36 2025] LNetError: 4070:0:(lib-ptl.c:189:lnet_try_match_md()) Skipped 1 previous similar message
```

## Run

The following submission script is configured to use 1 node with up to 2 tasks.

```bash
sbatch sbatch_dd_repro.sh
```
