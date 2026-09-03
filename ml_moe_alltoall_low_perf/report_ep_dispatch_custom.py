#!/usr/bin/env python3
r"""Aggregate and report custom EP dispatch benchmark results.

Reads per-rank JSON files written by ep_dispatch_bench_custom.py (one per chunk
of a sweep), computes per-node and per-chunk dispatch/combine medians, and flags
chunks or nodes that are much slower than their peers.

Usage:
    python3 report_ep_dispatch_custom.py --outdir results-ep-custom-3151234
    python3 report_ep_dispatch_custom.py --outdir 'results-ep-custom-*'
    python3 report_ep_dispatch_custom.py --outdir results-ep-custom-3151234 \
                                         --outdir results-ep-custom-3151235 \
                                         --csv ep-custom.csv
"""

import argparse
import collections
import csv
import json
import os
import pathlib
import re
import statistics
import sys
from typing import Dict, List, Optional

SLOW = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="summarize custom EP dispatch benchmark")
    p.add_argument("--outdir", type=str, required=True, action="append",
                   help="result directory (may be given multiple times or as a glob)")
    p.add_argument("--csv", type=str, default=None,
                   help="optional CSV output path")
    p.add_argument("--slow-threshold", type=float, default=SLOW,
                   help="multiplier above median used to flag slow chunks/nodes")
    return p.parse_args()


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def jobid_from_dir(d: pathlib.Path) -> str:
    """Extract a numeric jobid from the result directory name if present."""
    m = re.search(r'(\d{6,})$', d.name)
    return m.group(1) if m else d.name


def load_rank_file(path: pathlib.Path, outdir: pathlib.Path) -> Optional[Dict]:
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cfg = d.get("config", {})
    disp = d.get("dispatch_us", [])
    comb = d.get("combine_us", [])
    if not disp:
        return None
    return {
        "rank": d["rank"],
        "host": d["host"],
        "local_rank": d["local_rank"],
        "group_id": d["group_id"],
        "chunk_id": d.get("chunk_id", 0),
        "total_chunks": d.get("total_chunks", 1),
        "jobid": jobid_from_dir(outdir),
        "ep": cfg.get("ep", d.get("ep", 0)),
        "num_tokens": cfg.get("num_tokens", 0),
        "hidden": cfg.get("hidden", 0),
        "balanced": cfg.get("balanced_routing", False),
        "d_med": statistics.median(disp),
        "d_p90": pct(disp, 0.90),
        "d_max": max(disp),
        "c_med": statistics.median(comb) if comb else float("nan"),
        "c_p90": pct(comb, 0.90) if comb else float("nan"),
        "per_node_tokens": d.get("per_node_tokens", []),
    }


def summarize_chunks(rows: List[Dict], slow_threshold: float):
    by_chunk = collections.defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)

    chunk_summaries = []
    for chunk_id, chunk_rows in sorted(by_chunk.items()):
        d_meds = [r["d_med"] for r in chunk_rows]
        chunk_summaries.append({
            "chunk_id": chunk_id,
            "jobids": sorted({r["jobid"] for r in chunk_rows}),
            "nodes": sorted({r["host"] for r in chunk_rows}),
            "d_med": statistics.median(d_meds),
            "d_min": min(d_meds),
            "d_max": max(d_meds),
            "d_spread": max(d_meds) / min(d_meds) if min(d_meds) > 0 else float("inf"),
            "c_med": statistics.median([r["c_med"] for r in chunk_rows if r["c_med"] == r["c_med"]])
                       if any(r["c_med"] == r["c_med"] for r in chunk_rows) else float("nan"),
            "n_ranks": len(chunk_rows),
        })

    overall_d_med = statistics.median([c["d_med"] for c in chunk_summaries])

    print(f"\n=== per chunk (median dispatch across ranks) -- worst first, '<<' = > {slow_threshold:.1f}x overall median ({overall_d_med:.0f} us) ===")
    print(f"{'chunk':>6} {'jobid':>10} {'d_med':>8} {'d_min':>8} {'d_max':>8} {'spread':>7} {'c_med':>8}  nodes")
    flagged_chunks = []
    for c in sorted(chunk_summaries, key=lambda x: -x["d_med"]):
        slow = c["d_med"] > slow_threshold * overall_d_med
        if slow:
            flagged_chunks.append(c)
        short = ",".join(n.replace("nid00", "") for n in c["nodes"])
        jobids = ",".join(c["jobids"])
        print(f"{c['chunk_id']:>6} {jobids:>10} {c['d_med']:>8.0f} {c['d_min']:>8.0f} {c['d_max']:>8.0f} "
              f"{c['d_spread']:>7.2f} {c['c_med']:>8.0f}  {short}{'  <<' if slow else ''}")

    return chunk_summaries, flagged_chunks, overall_d_med


def summarize_nodes(rows: List[Dict], slow_threshold: float, overall_d_med: float):
    by_node = collections.defaultdict(list)
    for r in rows:
        by_node[r["host"]].append(r)

    node_summaries = []
    for host, node_rows in by_node.items():
        d_meds = [r["d_med"] for r in node_rows]
        node_summaries.append({
            "host": host,
            "chunks": sorted({r["chunk_id"] for r in node_rows}),
            "jobids": sorted({r["jobid"] for r in node_rows}),
            "d_med": statistics.median(d_meds),
            "d_min": min(d_meds),
            "d_max": max(d_meds),
            "n_ranks": len(node_rows),
        })

    print(f"\n=== per node (median dispatch across all chunk appearances) -- worst first ===")
    print(f"{'node':>12} {'d_med':>8} {'d_min':>8} {'d_max':>8} {'chunks':>16} {'jobids':>16} {'ranks':>6}")
    flagged_nodes = []
    for n in sorted(node_summaries, key=lambda x: -x["d_med"]):
        slow = n["d_med"] > slow_threshold * overall_d_med
        if slow:
            flagged_nodes.append(n)
        chunks = ",".join(str(c) for c in n["chunks"])
        jobids = ",".join(n["jobids"])
        print(f"{n['host']:>12} {n['d_med']:>8.0f} {n['d_min']:>8.0f} {n['d_max']:>8.0f} "
              f"{chunks:>16} {jobids:>16} {n['n_ranks']:>6}{'  <<' if slow else ''}")

    return node_summaries, flagged_nodes


def main() -> int:
    args = parse_args()
    outdirs: List[pathlib.Path] = []
    for raw in args.outdir:
        # Expand globs; if the shell did not expand them, do it here.
        expanded = sorted(pathlib.Path.cwd().glob(raw)) if '*' in raw or '?' in raw else [pathlib.Path(raw)]
        for d in expanded:
            if d.is_dir():
                outdirs.append(d)
            else:
                print(f"warning: {d} is not a directory, skipping", file=sys.stderr)

    if not outdirs:
        print("no valid result directories found", file=sys.stderr)
        return 1

    rows = []
    for d in outdirs:
        rows.extend(r for r in (load_rank_file(p, d) for p in d.glob("ep_custom_chunk*_rank*.json")) if r)

    if not rows:
        print(f"no rank result files found in {', '.join(str(d) for d in outdirs)}", file=sys.stderr)
        return 1

    shapes = {(r["num_tokens"], r["hidden"], r["ep"], r["balanced"]) for r in rows}
    print(f"Loaded {len(rows)} rank results from {', '.join(str(d) for d in outdirs)}")
    print(f"Shapes seen: " + "  ".join(
        f"tokens={t} hidden={h} ep={e} balanced={b}" for t, h, e, b in sorted(shapes)
    ))

    chunk_summaries, flagged_chunks, overall_d_med = summarize_chunks(rows, args.slow_threshold)
    _, flagged_nodes = summarize_nodes(rows, args.slow_threshold, overall_d_med)

    print(f"\nOverall median dispatch across chunks: {overall_d_med:.0f} us")
    if flagged_chunks:
        print(f"Flagged chunks: {len(flagged_chunks)}")
    if flagged_nodes:
        print(f"Flagged nodes: {len(flagged_nodes)}")
        for n in flagged_nodes:
            print(f"  {n['host']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["jobid", "chunk_id", "node", "rank", "local_rank", "d_med_us",
                        "d_p90_us", "d_max_us", "c_med_us", "c_p90_us"])
            for r in sorted(rows, key=lambda x: (x["chunk_id"], x["rank"])):
                w.writerow([
                    r["jobid"], r["chunk_id"], r["host"], r["rank"], r["local_rank"],
                    f"{r['d_med']:.1f}", f"{r['d_p90']:.1f}", f"{r['d_max']:.1f}",
                    f"{r['c_med']:.1f}" if r["c_med"] == r["c_med"] else "",
                    f"{r['c_p90']:.1f}" if r["c_p90"] == r["c_p90"] else "",
                ])
        print(f"WROTE {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
