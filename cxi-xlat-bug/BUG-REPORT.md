# Bug Report: `pri_mb_xlat_pf_addr_error` causes elbencho/libfabric CXI benchmark crash

**Reporter:** Jonathan Coles, CSCS  
**Date:** 2026-11-03  
**Severity:** High – user application crash / loss of network connectivity  
**Component:** HPE Slingshot CXI driver / Cassini NIC  
**Reproducer:** https://github.com/eth-cscs/alps-gh200-reproducers/tree/main/cxi-xlat-bug

---

## Summary

While running the `elbencho` network benchmark over libfabric/CXI on Alps GH200 nodes, the benchmark reproducibly crashes when all four Cassini NICs (`cxi[0-3]`) are used with 32 threads. Immediately before the crash, the CXI driver logs:

```
cxi_ss1 0020:01:00.0: cxi2[hsn3]: C_EC_INFO: C_PI_EXT error: pri_mb_xlat_pf_addr_error (12) (was first error at 1793280861:890077994)
```

This error has been previously characterized by HPE as "log noise", but in this case it is correlated with a real failure: the benchmark stops making progress, client worker threads report "Timed out waiting for response from server", and the job step must be killed. Reducing the thread count or the number of active NICs avoids the failure.

---

## Environment

| Item | Value |
|------|-------|
| System | Alps GH200 (CSCS) |
| OS / kernel | SUSE Linux, `6.4.0-150600.23.143-64kb` aarch64 |
| SHS stack | 13.1.0 (`shs-cassini-headers`, `shs-cxi-driver`, `shs-libcxi`) |
| Network provider | `cxi` |
| Network engine | `libfabric` |
| Benchmark | `elbencho` (built from `breuner/elbencho`, gcc-14, `LIBFABRIC_SUPPORT=1 BUILD_LIBFABRIC=1`) |
| NICs per node | 4 (`cxi0`, `cxi1`, `cxi2`, `cxi3`) |
| Nodes used | 2 for main reproducer (1 server + 1 client); 3 for two-client tests |

The `elbencho` binary is built inside a container using the included `Dockerfile` and exported to the host:

```bash
podman build --target export --output ./ -t elbencho:v1 -f ./Dockerfile
```

---

## Steps to Reproduce

1. Build the reproducer:
   ```bash
   cd cxi-xlat-bug
   podman build --target export --output ./ -t elbencho:v1 -f ./Dockerfile
   ```

2. Submit the Slurm job:
   ```bash
   sbatch submit.sh
   ```

3. The submission script runs a sweep of thread counts and device selections. The failing invocation is:
   ```bash
   ./elbencho --label "cxi[0-3],32" --resfile results-${SLURM_JOBID}.txt \
              --netprov cxi --netengine libfabric --netbench \
              --servers <server> --clients <client> -t 32 \
              --sendsize 1M --recvsize 0 \
              --timelimit 150 --base10 --graph --live1n
   ```

---

## Observed Behavior

### Main results (`results-89635.txt`, job 89635)

Results files are best viewed with `less -r`.

| Devices | Threads | Result | Approx. throughput |
|---------|---------|--------|-------------------|
| `cxi[0-3]` | 1 | PASS | ~7 GB/s |
| `cxi[0-3]` | 2 | PASS | ~7.9 GB/s |
| `cxi[0-3]` | 4 | PASS | ~13.9 GB/s |
| `cxi[0-3]` | 8 | PASS | ~24.6 GB/s |
| `cxi[0-3]` | 16 | PASS | ~54.2 GB/s |
| `cxi[0-3]` | 32 | **FAIL** | benchmark hangs/crashes after ~20–22 s |
| `cxi[0-1]` | 32 | PASS | ~48.4 GB/s |
| `cxi[2-3]` | 32 | PASS | ~48.4 GB/s |

### Failure signature

The benchmark begins normally, reaches high throughput, then stalls. Client-side worker threads report:

```
ERROR: Timed out waiting for response from server. Timeout: 20s; rounds done: ...
server: nid007680:1611; server endpoint: cxi FI_PROTO_CXI: ... (cxi0)
own endpoint: cxi FI_PROTO_CXI: ... (cxi0) (own interface cxi0); WorkerRank: ...
```

The server process continues to report receive progress, but the client side is no longer able to complete rounds. The Slurm step is eventually cancelled with `SIGNAL Killed`.

Representative excerpt from `slurm-89635.out` (step 89635.4, `cxi[0-3]`, 32 threads):

```
NET/libfabric: 71317 IOPS; 74781 MB/s; 811539 MiB; 64 threads; 2% CPU; 12s
NET/libfabric: 66244 IOPS; 69461 MB/s; 943961 MiB; 64 threads; 2% CPU; 14s
NET/libfabric: 71317 IOPS; 74781 MB/s; 1086525 MiB; 64 threads; 2% CPU; 16s
NET/libfabric: 69728 IOPS; 73115 MB/s; 1225982 MiB; 64 threads; 2% CPU; 18s
NET/libfabric: 64696 IOPS; 67838 MB/s; 1355374 MiB; 64 threads; 2% CPU; 20s
1: 14:49:32.155 ERROR: Timed out waiting for response from server. Timeout: 20s; ...
...
0: NET/libfabr Elapsed time     :     21.938s     22.153s
0:             IOPS recv        :       66273       65629
0:             MB/s recv        :       69492       68818
...
error: *** STEP 89635.4 ON nid007680 CANCELLED AT 2026-11-03T14:49:09 DUE TO SIGNAL Killed ***
```

The same failure is seen in other job runs, e.g. job 89634 (`slurm-89634.out`) and the two-client run in job 89303 (`slurm-89303.out`).

### Correlated kernel message

Immediately preceding the benchmark crash, the CXI driver logs:

```
cxi_ss1 0020:01:00.0: cxi2[hsn3]: C_EC_INFO: C_PI_EXT error: pri_mb_xlat_pf_addr_error (12) (was first error at 1793280861:890077994)
```

This message appears in the node kernel log (`dmesg`/journal) around the time of the hang. It is not present in the user-visible `slurm-*.out` files because it is emitted by the kernel driver.

---

## Expected Behavior

The benchmark should complete the full 150 s run without stalling or crashing when using all four NICs with 32 threads, just as it does with fewer threads or with two NICs and 32 threads.

---

## Key Findings / Isolation

- **Thread count matters:** With all four NICs, 16 threads is stable; 32 threads fails.
- **NIC count matters:** With 32 threads, using two NICs (`cxi[0-1]` or `cxi[2-3]`) is stable; using four NICs fails.
- **Failure is specific to the high-concurrency / all-NIC configuration.** This suggests a resource exhaustion or concurrency issue in the CXI driver/hardware translation path when all four NICs are heavily loaded.
- The `pri_mb_xlat_pf_addr_error` error is not benign log noise; it coincides with the loss of forward progress and termination of the user application.

---

## Supporting Files

All logs and results are included in the reproducer directory:

- `submit.sh` – Slurm submission script used to generate the results.
- `Dockerfile` – container build for the `elbencho` binary.
- `results-89635.txt` – main single-client sweep results.
- `results-89634.txt` – additional single-client run (also fails on `cxi[0-3]`, 32 threads).
- `results-89303.txt` – two-client sweep results (fails on `cxi[0-3]`, 32 threads).
- `slurm-89635.out`, `slurm-89634.out`, `slurm-89303.out` – full Slurm job output for the failing runs.

---
