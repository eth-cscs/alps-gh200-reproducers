#!/usr/bin/env python3
"""Plot dispatch/p90/max per-rank across log files, sorted by jobid."""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_PATTERN = re.compile(r".*-(\d+)\.out$")

RANK_RE = re.compile(r"^\s*(\d+)\s+\S+\s+\d+\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)")
HEADER_RE = re.compile(r"EP dispatch benchmark \| backend (\S+)\s+world (\d+)\s+nodes (\d+)\s+ranks/node (\d+)\s+EP (\d+)\s+groups (\d+)")
CONFIG_RE = re.compile(r"tokens (\d+)\s+hidden (\d+)\s+topk (\d+)\s+experts (\d+)\s+dtype (\S+)\s+payload (\d+) B/token")


def parse_summary_block(path: Path) -> dict:
    text = path.read_text()
    lines = text.splitlines()

    # Find the last summary table.
    header_idx = None
    for i, line in enumerate(lines):
        if "per rank (us)" in line:
            header_idx = i

    if header_idx is None:
        raise ValueError(f"No 'per rank (us)' table found in {path}")

    rows = []
    for line in lines[header_idx + 2:]:
        m = RANK_RE.match(line)
        if not m:
            break
        rank, dispatch, p90, max_ = m.groups()
        rows.append({
            "rank": int(rank),
            "dispatch": float(dispatch),
            "p90": float(p90),
            "max": float(max_),
        })

    if len(rows) != 8:
        raise ValueError(f"Expected 8 ranks in {path}, got {len(rows)}")

    meta = {}
    for line in lines:
        m = HEADER_RE.search(line)
        if m:
            meta.update({
                "backend": m.group(1),
                "world": int(m.group(2)),
                "nodes": int(m.group(3)),
                "ranks_per_node": int(m.group(4)),
                "EP": int(m.group(5)),
                "groups": int(m.group(6)),
            })
        m = CONFIG_RE.search(line)
        if m:
            meta.update({
                "tokens": int(m.group(1)),
                "hidden": int(m.group(2)),
                "topk": int(m.group(3)),
                "experts": int(m.group(4)),
                "dtype": m.group(5),
                "payload": int(m.group(6)),
            })

    return {"rows": rows, "meta": meta}


def load_data(file_glob: str) -> tuple[pd.DataFrame, dict, list]:
    paths = [Path(p) for p in glob.glob(file_glob, recursive=True)]

    files = []
    for path in paths:
        if not path.is_file():
            continue
        m = OUT_PATTERN.match(path.name)
        if not m:
            # Fallback: try to extract the last numeric run in the stem.
            nums = re.findall(r"\d+", path.stem)
            if not nums:
                continue
            jobid = int(nums[-1])
        else:
            jobid = int(m.group(1))
        files.append((jobid, path))

    if not files:
        raise FileNotFoundError(f"No matching .out files found for glob: {file_glob}")

    files.sort(key=lambda x: x[0])

    records = []
    first_meta = None
    for jobid, path in files:
        parsed = parse_summary_block(path)
        if first_meta is None:
            first_meta = parsed["meta"]
        for row in parsed["rows"]:
            records.append({
                "jobid": jobid,
                "file": path.name,
                "rank": row["rank"],
                "dispatch": row["dispatch"],
                "p90": row["p90"],
                "max": row["max"],
            })

    df = pd.DataFrame(records)
    df["index"] = df.groupby("jobid").ngroup()
    return df, first_meta, files


def plot(df: pd.DataFrame, meta: dict, files: list, output: str, file_glob: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        "EP dispatch benchmark: per-rank timing across jobs",
        fontsize=14,
        fontweight="bold",
    )

    metrics = ["dispatch", "p90", "max"]
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "v", "<", ">", "d", "p"]

    for ax, metric in zip(axes, metrics):
        for rank in range(8):
            sub = df[df["rank"] == rank]
            ax.plot(
                sub["index"],
                sub[metric],
                marker=markers[rank],
                color=colors[rank],
                markersize=4,
                linewidth=1.2,
                label=f"rank {rank}",
            )
        ax.set_ylabel(f"{metric} (us)")
        ax.set_title(f"{metric}")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("job index (sorted by jobid)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        ncol=1,
        frameon=True,
        title="rank",
    )

    n = len(files)
    first_job = files[0][0]
    last_job = files[-1][0]
    meta_text = (
        f"glob: {file_glob}\n"
        f"backend: {meta.get('backend', 'N/A')}\n"
        f"world: {meta.get('world', 'N/A')}  nodes: {meta.get('nodes', 'N/A')}  ranks/node: {meta.get('ranks_per_node', 'N/A')}\n"
        f"EP: {meta.get('EP', 'N/A')}  groups: {meta.get('groups', 'N/A')}\n"
        f"tokens: {meta.get('tokens', 'N/A')}  hidden: {meta.get('hidden', 'N/A')}  topk: {meta.get('topk', 'N/A')}  experts: {meta.get('experts', 'N/A')}\n"
        f"dtype: {meta.get('dtype', 'N/A')}  payload: {meta.get('payload', 'N/A')} B/token\n"
        f"files: {n}  jobid range: {first_job} .. {last_job}\n"
        "x-axis: sequential index after sorting by jobid"
    )

    fig.text(
        0.02,
        0.01,
        meta_text,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0.12, 0.85, 0.96])
    plt.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-rank dispatch/p90/max from summary blocks of .out files."
    )
    parser.add_argument(
        "--glob",
        default="logs/*.out",
        help="Glob pattern for .out files to parse (default: logs/*.out).",
    )
    parser.add_argument(
        "-o", "--output",
        default="dispatch_summary.pdf",
        help="Output image filename (default: dispatch_summary.pdf).",
    )
    args = parser.parse_args()

    df, meta, files = load_data(args.glob)
    print(df.head(16))
    print(f"\nLoaded {len(files)} files, {len(df)} rows")
    plot(df, meta, files, args.output, args.glob)


if __name__ == "__main__":
    main()
