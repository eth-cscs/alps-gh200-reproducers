// SPDX-License-Identifier: GPL-2.0-only
/*
 * wcprobe - CPU accesses to a CXI command-queue launch page (Grace/arm64)
 *
 * Allocates one transmit and one target command queue through libcxi (as
 * libfabric does) and accesses the launch page of one of them:
 *
 *   write  stores only: CQ NOOP commands via doorbell and low-latency writes
 *   prfm   PRFM PLDL1KEEP hints on launch-page lines
 *   spec   bounds-check-bypass gadget: a mispredicted branch makes the CPU
 *          speculatively load from the launch page; no load to the launch
 *          page is ever architecturally executed
 *   read   one real load (--width 8 or 64) from the launch page
 *
 * Accesses go to line (--offset + k * 64) % PAGE_SIZE, k = 0 .. --lines-1, so
 * the address in C_PI_ERR_INFO_RTRGT can be matched: TXQ n launch page is at
 * BAR0 + n * 64K, TGT n at BAR0 + 64M + n * 64K. The program prints n and the
 * expected BAR0 offset of line 0.
 *
 * Build: gcc -O2 -o wcprobe wcprobe.c $(pkg-config --cflags --libs libcxi)
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <sched.h>
#include <stdarg.h>
#include <getopt.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <libcxi.h>

#if !defined(__aarch64__)
#error "wcprobe targets aarch64 (Grace)"
#endif

static const char *mode = "write";
static unsigned int dev_id = 0, iters = 100000, lines = 4, offset = 0, use_tgt = 0;
static unsigned int read_width = 64;
static long page;

static double now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...)
{
	va_list ap;
	printf("[%.6f] ", now());
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	putchar('\n');
	fflush(stdout);
}

static sigjmp_buf jb;
static volatile sig_atomic_t got_sig;
static void on_sig(int sig, siginfo_t *si, void *uc)
{
	(void)uc;
	got_sig = sig;
	fprintf(stdout, "[%.6f] SIGNAL %d (%s) addr=%p code=%d\n", now(), sig,
		strsignal(sig), si->si_addr, si->si_code);
	fflush(stdout);
	siglongjmp(jb, 1);
}

static inline void *line_addr(uint8_t *base, unsigned int k)
{
	return base + ((offset + (size_t)k * 64) % (size_t)page);
}

/* ---- write: stores only (doorbell + low-latency path) ---- */
static int do_write(struct cxi_cq *cq)
{
	for (unsigned int i = 0; i < iters; i++) {
		if (cxi_cq_emit_cq_cmd(cq, C_CMD_CQ_NOOP)) {
			cxi_cq_ring(cq);
			continue;
		}
		if (i & 1)
			cxi_cq_ll_ring(cq);
		else
			cxi_cq_ring(cq);
	}
	cxi_cq_ring(cq);
	return 0;
}

/* ---- prfm: software prefetch hints ---- */
static int do_prfm(uint8_t *la)
{
	for (unsigned int i = 0; i < iters; i++)
		for (unsigned int k = 0; k < lines; k++)
			asm volatile("prfm pldl1keep, [%0]" :: "r"(line_addr(la, k)) : "memory");
	return 0;
}

/* ---- spec: bounds-check-bypass gadget ----
 * ptrs[0] = harmless host buffer, ptrs[1] = launch page line.
 * bound = 1 lives in its own cache line and is flushed before every attempt so
 * the branch resolves late; the predictor is trained with idx=0 (in bounds) 15
 * times, then called with idx=1 (out of bounds): architecturally skipped,
 * speculatively executed. */
static uint8_t safe_buf[4096] __attribute__((aligned(4096)));
static struct { volatile uint64_t bound; uint8_t pad[248]; } bnd __attribute__((aligned(256))) = { .bound = 1 };
static uint8_t *volatile ptrs[2];
static volatile uint64_t sink;

static __attribute__((noinline)) void gadget(unsigned int idx)
{
	if (idx < bnd.bound)
		sink += *(volatile uint64_t *)ptrs[idx];
}

static int do_spec(uint8_t *la)
{
	ptrs[0] = safe_buf;
	unsigned long attempts = 0;
	uint64_t x = 0x9e3779b97f4a7c15ULL;
	for (unsigned int i = 0; i < iters; i++) {
		ptrs[1] = line_addr(la, i % lines);
		for (int t = 0; t < 16; t++) {
			/* pseudo-random ~1/16 out-of-bounds calls, so the predictor
			 * cannot learn the pattern; idx computed without a branch */
			x ^= x << 13; x ^= x >> 7; x ^= x << 17;
			unsigned int idx = ((x & 15) == 0);
			attempts += idx;
			asm volatile("dc civac, %0\n\tdsb ish" :: "r"(&bnd.bound) : "memory");
			gadget(idx);
		}
	}
	say("spec: %lu mispredicted attempts, sink=%lu", attempts, (unsigned long)sink);
	return 0;
}

/* ---- read: real loads (positive control) ---- */
static int do_read(uint8_t *la)
{
	struct sigaction sa = { .sa_sigaction = on_sig, .sa_flags = SA_SIGINFO };
	sigaction(SIGBUS, &sa, NULL);
	sigaction(SIGSEGV, &sa, NULL);
	unsigned int n = iters < lines ? iters : lines; /* one load per line */
	for (unsigned int k = 0; k < n; k++) {
		void *a = line_addr(la, k);
		say("read: %u-byte load at launch+0x%lx", read_width,
		    (unsigned long)((uint8_t *)a - la));
		sleep(1); /* let the log line out before a possible SError */
		if (sigsetjmp(jb, 1)) {
			say("read: load faulted (signal %d), continuing", (int)got_sig);
			continue;
		}
		if (read_width == 64)
			asm volatile("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%0]"
				     :: "r"(a) : "v0", "v1", "v2", "v3", "memory");
		else
			sink += *(volatile uint64_t *)a;
		say("read: load returned");
	}
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
		"usage: wcprobe --mode write|prfm|spec|read [--dev N] [--tgt] [--iters N]\n"
		"               [--lines N] [--offset BYTES] [--width 8|64]\n");
	exit(2);
}

int main(int argc, char **argv)
{
	static const struct option o[] = {
		{ "mode", 1, 0, 'm' }, { "dev", 1, 0, 'd' }, { "tgt", 0, 0, 't' },
		{ "iters", 1, 0, 'i' }, { "lines", 1, 0, 'l' }, { "offset", 1, 0, 'o' },
		{ "width", 1, 0, 'w' }, { 0, 0, 0, 0 } };
	int c;
	while ((c = getopt_long(argc, argv, "", o, NULL)) != -1) {
		switch (c) {
		case 'm': mode = optarg; break;
		case 'd': dev_id = strtoul(optarg, 0, 0); break;
		case 't': use_tgt = 1; break;
		case 'i': iters = strtoul(optarg, 0, 0); break;
		case 'l': lines = strtoul(optarg, 0, 0); break;
		case 'o': offset = strtoul(optarg, 0, 0); break;
		case 'w': read_width = strtoul(optarg, 0, 0); break;
		default: usage();
		}
	}
	if (!lines || (read_width != 8 && read_width != 64) || (offset & 63))
		usage();
	page = sysconf(_SC_PAGESIZE);

	struct cxil_dev *dev;
	struct cxil_lni *lni;
	struct cxi_cp *cp;
	struct cxi_cq *txq, *tgq;
	int rc;
	if ((rc = cxil_open_device(dev_id, &dev))) { say("cxil_open_device(%u): %s", dev_id, strerror(-rc)); return 1; }
	if ((rc = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID))) { say("cxil_alloc_lni: %s", strerror(-rc)); return 1; }
	if ((rc = cxil_alloc_cp(lni, 1, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT, &cp))) { say("cxil_alloc_cp: %s", strerror(-rc)); return 1; }
	struct cxi_cq_alloc_opts opts = { .count = 1024, .flags = CXI_CQ_IS_TX, .lcid = cp->lcid };
	if ((rc = cxil_alloc_cmdq(lni, NULL, &opts, &txq))) { say("alloc TX cmdq: %s", strerror(-rc)); return 1; }
	opts.flags = 0;
	if ((rc = cxil_alloc_cmdq(lni, NULL, &opts, &tgq))) { say("alloc TGT cmdq: %s", strerror(-rc)); return 1; }

	struct cxi_cq *cq = use_tgt ? tgq : txq;
	uint8_t *la = (uint8_t *)cq->wp_addr;
	say("pid=%d cpu=%d dev=cxi%u page=%ld mode=%s iters=%u lines=%u offset=0x%x width=%u",
	    getpid(), sched_getcpu(), dev_id, page, mode, iters, lines, offset, read_width);
	say("TXQ cqn=%u launch=%p | TGT cqn=%u launch=%p | using %s",
	    cxi_cq_get_cqn(txq), (void *)txq->wp_addr, cxi_cq_get_cqn(tgq),
	    (void *)tgq->wp_addr, use_tgt ? "TGT" : "TXQ");
	say("expected dmesg BAR0 offset of line 0: 0x%lx",
	    (unsigned long)((use_tgt ? 0x4000000UL : 0) + cxi_cq_get_cqn(cq) * 0x10000UL +
			    (offset % page)));

	double t0 = now();
	say("BEGIN %s", mode);
	if (!strcmp(mode, "write")) rc = do_write(cq);
	else if (!strcmp(mode, "prfm")) rc = do_prfm(la);
	else if (!strcmp(mode, "spec")) rc = do_spec(la);
	else if (!strcmp(mode, "read")) rc = do_read(la);
	else usage();
	say("END %s rc=%d (%.3f s)", mode, rc, now() - t0);

	cxil_destroy_cmdq(tgq);
	cxil_destroy_cmdq(txq);
	cxil_destroy_cp(cp);
	cxil_destroy_lni(lni);
	cxil_close_device(dev);
	return rc;
}
