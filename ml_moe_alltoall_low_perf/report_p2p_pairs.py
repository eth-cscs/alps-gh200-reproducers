#!/usr/bin/env python3
r"""Summarize point-to-point pair latencies from the p2p benchmark variant.

Reads the aggregated JSON files written by ep_dispatch_bench_mpi_cpu_p2p
(usually named ep-dispatch-<jobid>.json under the result directory) and prints
per-pair min/median/max statistics, along with flags for slow or highly
variable pairs.

Usage:
    python3 report_p2p_pairs.py --outdir results-ep-mpi-cpu-ctn-3151234
    python3 report_p2p_pairs.py --outdir 'results-ep-mpi-cpu-*'
    python3 report_p2p_pairs.py --outdir results-ep-mpi-cpu-ctn-3151234 \
                                --csv p2p-pairs.csv
"""

import argparse
import csv
import json
import pathlib
import statistics
import sys
from typing import Dict, List, Optional, Tuple

SLOW = 1.5
SPREAD = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="summarize p2p pair latencies")
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        action="append",
        help="result directory (may be given multiple times or as a glob)",
    )
    p.add_argument(
        "--csv", type=str, default=None, help="optional CSV output path"
    )
    p.add_argument(
        "--slow-threshold",
        type=float,
        default=SLOW,
        help="multiplier above median pair latency used to flag slow pairs",
    )
    p.add_argument(
        "--spread-threshold",
        type=float,
        default=SPREAD,
        help="multiplier between max and min latency used to flag variable pairs",
    )
    p.add_argument(
        "--worst-n",
        type=int,
        default=20,
        help="number of worst pairs to print in the summary tables",
    )
    return p.parse_args()


def load_agg_file(path: pathlib.Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_rank_p2p(path: pathlib.Path) -> Optional[Dict]:
    """Load p2p raw samples from a per-rank JSON file."""
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cfg = d.get("config", {})
    pairs = d.get("p2p_pairs", [])
    if not pairs:
        return None
    rank = d.get("rank", 0)
    host = d.get("host", "")
    pair_bytes = d.get("p2p_pair_bytes", 0)
    p2p_iters = d.get("p2p_iters", 0)
    records = []
    for pair in pairs:
        dst = pair.get("dst")
        us = pair.get("us", [])
        if dst is None or not us:
            continue
        records.append({
            "src": rank,
            "src_host": host,
            "dst": dst,
            "pair_bytes": pair_bytes,
            "min_us": min(us),
            "max_us": max(us),
            "median_us": statistics.median(us),
            "iters": p2p_iters,
            "num_tokens": cfg.get("num_tokens", 0),
            "hidden": cfg.get("hidden", 0),
            "ep": cfg.get("ep", 0),
            "balanced": cfg.get("balanced_routing", False),
        })
    return {"records": records, "cfg": cfg} if records else None


def summarize(outdirs: List[pathlib.Path], slow_threshold: float, spread_threshold: float):
    rows: List[Dict] = []
    shapes = set()
    for d in outdirs:
        # Map rank -> host from per-rank files, populated regardless of whether
        # we later use the aggregated summary or per-rank raw data.
        rank_hosts: Dict[int, str] = {}
        for p in d.glob("ep_custom_chunk*_rank*.json"):
            try:
                rd = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rank_hosts[rd.get("rank", -1)] = rd.get("host", "")

        # First try aggregated JSON files with p2p_summary.
        found_summary = False
        for p in d.glob("ep-dispatch-*.json"):
            data = load_agg_file(p)
            if data is None:
                continue
            cfg = data.get("config", {})
            ranks = data.get("ranks", [])
            # Prefer the host field inside the aggregated ranks array; fall back
            # to the per-rank file map if the aggregated copy is incomplete.
            for rd in ranks:
                rank = rd.get("rank", -1)
                if rank >= 0:
                    rank_hosts[rank] = rd.get("host", rank_hosts.get(rank, ""))
            summary = data.get("p2p_summary")
            if summary:
                found_summary = True
                shapes.add(
                    (
                        cfg.get("num_tokens", 0),
                        cfg.get("hidden", 0),
                        cfg.get("ep", 0),
                        cfg.get("balanced_routing", False),
                        summary.get("pair_bytes", 0),
                    )
                )
                for pair in summary.get("pairs", []):
                    src = pair["src"]
                    dst = pair["dst"]
                    rows.append(
                        {
                            "jobid": p.stem.replace("ep-dispatch-", ""),
                            "outdir": str(p.parent),
                            "src": src,
                            "dst": dst,
                            "src_host": rank_hosts.get(src, ""),
                            "dst_host": rank_hosts.get(dst, ""),
                            "pair_bytes": summary.get("pair_bytes", 0),
                            "min_us": pair["min_us"],
                            "max_us": pair["max_us"],
                            "median_us": pair["median_us"],
                        }
                    )
        # Fallback to per-rank files if no valid aggregated p2p_summary was found.
        if not found_summary:
            for p in d.glob("ep_custom_chunk*_rank*.json"):
                loaded = load_rank_p2p(p)
                if loaded is None:
                    continue
                for r in loaded["records"]:
                    src = r["src"]
                    dst = r["dst"]
                    shapes.add(
                        (
                            r["num_tokens"],
                            r["hidden"],
                            r["ep"],
                            r["balanced"],
                            r["pair_bytes"],
                        )
                    )
                    rows.append(
                        {
                            "jobid": p.parent.name,
                            "outdir": str(p.parent),
                            "src": src,
                            "dst": dst,
                            "src_host": rank_hosts.get(src, ""),
                            "dst_host": rank_hosts.get(dst, ""),
                            "pair_bytes": r["pair_bytes"],
                            "min_us": r["min_us"],
                            "max_us": r["max_us"],
                            "median_us": r["median_us"],
                        }
                    )
    return rows, shapes


def main() -> int:
    args = parse_args()
    outdirs: List[pathlib.Path] = []
    for raw in args.outdir:
        expanded = (
            sorted(pathlib.Path.cwd().glob(raw))
            if "*" in raw or "?" in raw
            else [pathlib.Path(raw)]
        )
        for d in expanded:
            if d.is_dir():
                outdirs.append(d)
            else:
                print(f"warning: {d} is not a directory, skipping", file=sys.stderr)

    if not outdirs:
        print("no valid result directories found", file=sys.stderr)
        return 1

    rows, shapes = summarize(outdirs, args.slow_threshold, args.spread_threshold)
    if not rows:
        print(
            f"no p2p data found in {', '.join(str(d) for d in outdirs)}",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded {len(rows)} p2p pairs from {', '.join(str(d) for d in outdirs)}")
    print(
        "Shapes seen: "
        + "  ".join(
            f"tokens={t} hidden={h} ep={e} balanced={b} pair_bytes={pb}"
            for t, h, e, b, pb in sorted(shapes)
        )
    )

    overall_med = statistics.median([r["median_us"] for r in rows])
    overall_min = min(r["min_us"] for r in rows)
    overall_max = max(r["max_us"] for r in rows)

    for r in rows:
        r["same_node"] = bool(r["src_host"] and r["src_host"] == r["dst_host"])
        r["slow"] = r["median_us"] > args.slow_threshold * overall_med
        r["variable"] = (
            r["min_us"] > 0 and r["max_us"] / r["min_us"] > args.spread_threshold
        )
        r["spread"] = r["max_us"] / r["min_us"] if r["min_us"] > 0 else float("inf")

    print(f"\nOverall pair median: {overall_med:.0f} us")
    print(f"Overall min/max: {overall_min:.0f} us / {overall_max:.0f} us")
    print(f"Slow threshold: >{args.slow_threshold:.1f}x median ({args.slow_threshold * overall_med:.0f} us)")
    print(f"Variable threshold: max/min >{args.spread_threshold:.1f}x")

    by_node = {
        "same": [r for r in rows if r["same_node"]],
        "diff": [r for r in rows if not r["same_node"]],
    }
    for label, subset in by_node.items():
        if subset:
            med = statistics.median([r["median_us"] for r in subset])
            mn = min(r["min_us"] for r in subset)
            mx = max(r["max_us"] for r in subset)
            print(f"{label}-node pairs: {len(subset):>3}  median {med:>8.0f} us  min {mn:>8.0f} us  max {mx:>8.0f} us")

    print(f"\n=== slowest pairs by median latency (top {args.worst_n}) ===")
    print(f"{'src':>5} {'dst':>5} {'loc':>4} {'median_us':>11} {'min_us':>9} {'max_us':>9} {'spread':>7}  {'flags':>10}")
    for r in sorted(rows, key=lambda x: -x["median_us"])[: args.worst_n]:
        flags = []
        if r["slow"]:
            flags.append("SLOW")
        if r["variable"]:
            flags.append("VAR")
        loc = "same" if r["same_node"] else "diff"
        print(
            f"{r['src']:>5} {r['dst']:>5} {loc:>4} {r['median_us']:>11.1f} {r['min_us']:>9.1f} "
            f"{r['max_us']:>9.1f} {r['spread']:>7.2f}  {','.join(flags) or '-':>10}"
        )

    print(f"\n=== most variable pairs by max/min spread (top {args.worst_n}) ===")
    print(f"{'src':>5} {'dst':>5} {'loc':>4} {'median_us':>11} {'min_us':>9} {'max_us':>9} {'spread':>7}  {'flags':>10}")
    for r in sorted(rows, key=lambda x: -x["spread"])[: args.worst_n]:
        flags = []
        if r["slow"]:
            flags.append("SLOW")
        if r["variable"]:
            flags.append("VAR")
        loc = "same" if r["same_node"] else "diff"
        print(
            f"{r['src']:>5} {r['dst']:>5} {loc:>4} {r['median_us']:>11.1f} {r['min_us']:>9.1f} "
            f"{r['max_us']:>9.1f} {r['spread']:>7.2f}  {','.join(flags) or '-':>10}"
        )

    slow = [r for r in rows if r["slow"]]
    variable = [r for r in rows if r["variable"]]
    if slow:
        print(f"\nFlagged slow pairs: {len(slow)}")
    if variable:
        print(f"Flagged variable pairs: {len(variable)}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(
                [
                    "jobid",
                    "src",
                    "dst",
                    "src_host",
                    "dst_host",
                    "same_node",
                    "pair_bytes",
                    "min_us",
                    "median_us",
                    "max_us",
                    "spread",
                    "slow",
                    "variable",
                ]
            )
            for r in sorted(rows, key=lambda x: (-x["median_us"], x["src"], x["dst"])):
                w.writerow(
                    [
                        r["jobid"],
                        r["src"],
                        r["dst"],
                        r["src_host"],
                        r["dst_host"],
                        "1" if r["same_node"] else "0",
                        r["pair_bytes"],
                        f"{r['min_us']:.1f}",
                        f"{r['median_us']:.1f}",
                        f"{r['max_us']:.1f}",
                        f"{r['spread']:.2f}",
                        "1" if r["slow"] else "0",
                        "1" if r["variable"] else "0",
                    ]
                )
        print(f"WROTE {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
