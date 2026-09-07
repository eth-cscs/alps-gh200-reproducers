#!/usr/bin/env python3
r"""
Minimal EP dispatch/combine latency benchmark on UCCL -- no Megatron, no model.

Drives the same kernels tools/nsys_ep_latency.py extracts from training traces
(uccl::internode::{notify_dispatch,dispatch,cached_notify,combine}) with the
call shape and buffer sizing copied from megatron/.../moe/fused_a2a.py.

--backend nccl swaps those kernels for plain all_to_all_single on the same
routing, so a node's latency can be attributed to its NIC rather than to UCCL.

Defaults mirror chonk_kda_H3584_h2048_lt1792; `hidden` is the MoE latent, not
the model hidden size. Chunk tuning comes from UCCL_EP_{DISPATCH,COMBINE}_CONFIG
in the environment, so the launcher stays the only authority for it.

Launched by bench/ep_dispatch_bench.sbatch (sets RANK/LOCAL_RANK/WORLD_SIZE).
"""

import argparse
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist

torch.set_num_threads(1)          # Controls intra-op parallelism
torch.set_num_interop_threads(1)  # Controls inter-op parallelism


# Same threshold as tools/ep_slow_nodes.py, so "slow" means the same thing here.
SLOW = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EP dispatch latency benchmark")
    p.add_argument("--num-tokens", type=int, default=8192, help="tokens per rank (MBS x SEQ_LEN)")
    p.add_argument("--hidden", type=int, default=1792, help="dispatched hidden dim (MOE_LATENT_SIZE)")
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--ep", type=int, default=8, help="EP group size; WORLD_SIZE must be a multiple")
    p.add_argument("--backend", choices=("uccl", "nccl"), default="uccl",
                   help="uccl = DeepEP dispatch/combine; nccl = plain all_to_all_single")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--fp8", action="store_true", help="dispatch fp8 + scales (USE_FP8_DISPATCH=true)")
    p.add_argument("--no-combine", action="store_true", help="time dispatch only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write raw per-rank timings to this JSON path")
    return p.parse_args()


def pct(xs, q: float) -> float:
    """Nearest-rank percentile, to avoid a numpy import for six lines of stats."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def init_dist(ep: int):
    """Bring up NCCL from the Slurm env and cut the world into EP groups.

    EP groups are `ep` consecutive ranks (Megatron's tp-cp-ep-dp-pp order at
    TP=CP=1), so under block:block one EP32 group is 8 consecutive nodes.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world % ep:
        raise SystemExit(f"WORLD_SIZE={world} is not a multiple of --ep {ep}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", world_size=world, rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    # The reference UCCL tests set this before building the Buffer; kept for
    # fidelity with the known-good path. Every tensor below is explicit anyway.
    torch.set_default_dtype(torch.bfloat16)

    # new_group must be entered by every rank for every group, in the same order.
    my_group, my_group_id = None, rank // ep
    for gid in range(world // ep):
        g = dist.new_group(list(range(gid * ep, (gid + 1) * ep)))
        if gid == my_group_id:
            my_group = g
    return rank, world, local_rank, my_group, my_group_id


def make_buffer(group, hidden: int, elem_size: int):
    """Size the buffer the way megatron/core/transformer/moe/fused_a2a.py does."""
    from deep_ep import Buffer

    hidden_bytes = hidden * max(elem_size, 2)
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    # explicitly_destroy: releasing in the destructor can hang Python's exception
    # handling (see Buffer's docstring), and we want a clean exit either way.
    return Buffer(group, num_nvl_bytes, num_rdma_bytes, explicitly_destroy=True)


def time_iters(fn, iters: int, warmup: int) -> list:
    """Per-iteration us, each iteration started from a world-wide barrier.

    Without the barrier the ranks drift and each sample absorbs the previous
    iteration's skew; with it, one slow NIC slows its whole EP group visibly.
    """
    dist.barrier()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        #torch.cuda.synchronize()
        #torch.cuda.synchronize()
        dist.barrier()
        starts[i].record()
        fn()
        dist.barrier()
        ends[i].record()
    torch.cuda.synchronize()
    t= [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
    print(t)
    return t


def build_inputs(args, rank: int, device: str):
    """Uniform top-k over all experts: full mesh, so every node pair is loaded.

    Not the DeepEP reference's group-limited routing -- an untouched node pair
    cannot report a bad link.
    """
    torch.manual_seed(args.seed + rank)
    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
    # topk over random scores gives distinct experts per token; randint would not.
    topk_idx = torch.rand(
        (args.num_tokens, args.num_experts), dtype=torch.float32, device=device
    ).topk(args.num_topk, dim=-1).indices.to(torch.int64)
    topk_weights = torch.rand(
        (args.num_tokens, args.num_topk), dtype=torch.float32, device=device
    )
    if args.fp8:
        from deep_ep.utils import per_token_cast_to_fp8

        x_fp8, scales = per_token_cast_to_fp8(x)
        # DeepEP wants the scale tensor's last dim contiguous-transposed;
        # all_to_all_single splits along dim 0 and needs it row-major.
        if args.backend == "uccl":
            scales = scales.T.contiguous().T
        return (x_fp8, scales), topk_idx, topk_weights
    return x, topk_idx, topk_weights


def setup_uccl(args, group, x, topk_idx, topk_weights) -> dict:
    """DeepEP/UCCL internode dispatch + combine, the path training actually uses."""
    from deep_ep import Buffer
    from deep_ep.utils import per_token_cast_back

    elem_size = x[0].element_size() if isinstance(x, tuple) else x.element_size()
    buffer = make_buffer(group, args.hidden, elem_size)

    # Layout once: topk_idx is fixed and get_dispatch_layout is a local kernel
    # that the EP latency report excludes anyway.
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, args.num_experts)

    # handle stays None so every iteration pays notify_dispatch + the CPU sync,
    # exactly as the forward pass does.
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    def combine_setup():
        # One dispatch supplies the handle + payload, so the timed loop is
        # combine + cached_notify and nothing else.
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
        # combine takes bf16, and no topk_weights -- that is FusedCombine.forward.
        combine_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
        return lambda: buffer.combine(x=combine_x, handle=handle)

    def cfg_str(c):
        return ",".join(str(v) for v in (
            c.num_sms, c.num_max_nvl_chunked_send_tokens, c.num_max_nvl_chunked_recv_tokens,
            c.num_max_rdma_chunked_send_tokens, c.num_max_rdma_chunked_recv_tokens))

    return dict(
        dispatch=lambda: buffer.dispatch(**dispatch_args),
        combine_setup=combine_setup,
        # A token bound for a remote node crosses the NIC once (DeepEP sends to
        # the node, then forwards over NVLink), so tokens x payload = wire volume.
        per_node_tokens=num_tokens_per_rdma_rank.tolist(),
        desc=[f"dispatch cfg {cfg_str(Buffer.get_dispatch_config(args.ep))}   "
              f"combine cfg {cfg_str(Buffer.get_combine_config(args.ep))}",
              f"buffers: nvl {buffer.num_nvl_bytes / 1e9:.2f} GB  "
              f"rdma {buffer.num_rdma_bytes / 1e9:.2f} GB"],
        close=buffer.destroy,
    )


def setup_nccl(args, group, x, topk_idx, topk_weights, ranks_per_node: int) -> dict:
    """Dispatch as Megatron's non-DeepEP path does: exchange counts, sync on them,
    permute, then one all_to_all_single per payload tensor.

    Unlike DeepEP this has no node-level dedup, so a token wanted by ranks on two
    different remote nodes is what it is -- but within a rank it is sent once.
    """
    ep, device = group.size(), topk_idx.device
    # float8_e4m3fn goes over all_to_all_single as-is (ncclFloat8e4m3, NCCL >= 2.24).
    payload, scales = x if isinstance(x, tuple) else (x, None)

    # Destination rank set per token, deduplicated the way DeepEP's NVL side is.
    to_rank = torch.zeros((args.num_tokens, ep), dtype=torch.bool, device=device)
    to_rank.scatter_(1, topk_idx // (args.num_experts // ep), True)
    send_counts = to_rank.sum(0).to(torch.int32)
    # nonzero() on the transpose is sorted by destination, i.e. the send layout.
    send_idx = to_rank.t().nonzero()[:, 1].contiguous()

    print(send_counts)

    in_splits = send_counts.tolist()
    recv_counts = torch.empty_like(send_counts)

    def a2a(src, out_splits):
        out = torch.empty((sum(out_splits), src.shape[1]), dtype=src.dtype, device=device)
        #return []
        #out = torch.empty((sum(out_splits), src.shape[1]), dtype=src.dtype)
        dist.all_to_all_single(out, src, out_splits, in_splits, group=group)
        return out

    def dispatch():
        # The counts exchange plus the .tolist() sync is this path's notify_dispatch.
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        out_splits = recv_counts.tolist()
        recv = a2a(payload[send_idx], out_splits)
        return (recv, a2a(scales[send_idx], out_splits)) if scales is not None else recv
        #return recv

    def combine_setup():
        out_splits = recv_counts.tolist()  # cached, as cached_notify caches the handle
        n_recv = sum(out_splits)
        # combine is bf16 whatever dispatch sent, so only its volume differs.
        combine_x = torch.empty((n_recv, args.hidden), dtype=torch.bfloat16, device=device)

        def combine():
            back = torch.empty((send_idx.numel(), args.hidden), dtype=torch.bfloat16, device=device)
            dist.all_to_all_single(back, combine_x, in_splits, out_splits, group=group)
            out = torch.zeros((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
            return out.index_add_(0, send_idx, back)

        return combine

    # Prime recv_counts so combine_setup can size its buffers without a dispatch.
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    #out_splits = recv_counts.tolist()
    #src = payload[send_idx]
    #out = torch.empty((sum(out_splits), src.shape[1]), dtype=src.dtype, device=device)

    return dict(
        dispatch=dispatch,
        combine_setup=combine_setup,
        per_node_tokens=send_counts.view(-1, ranks_per_node).sum(1).tolist(),
        desc=[f"all_to_all_single over {ep} peers, "
              f"{sum(in_splits)} tokens sent ({sum(in_splits) / args.num_tokens:.2f}x replication)"],
        close=lambda: None,
    )


def main() -> int:
    args = parse_args()
    rank, world, local_rank, group, group_id = init_dist(args.ep)
    device = f"cuda:{local_rank}"

    if args.num_experts % args.ep:
        raise SystemExit(f"--num-experts {args.num_experts} not divisible by --ep {args.ep}")

    x, topk_idx, topk_weights = build_inputs(args, rank, device)
    if args.backend == "uccl":
        be = setup_uccl(args, group, x, topk_idx, topk_weights)
    else:
        rpn = int(os.environ.get("SLURM_NTASKS_PER_NODE") or torch.cuda.device_count())
        be = setup_nccl(args, group, x, topk_idx, topk_weights, rpn)

    dispatch_us = time_iters(be["dispatch"], args.iters, args.warmup)

    combine_us = []
    combine = None
    if not args.no_combine:
        combine = be["combine_setup"]()
        combine_us = time_iters(combine, args.iters, args.warmup)

    payload_bytes = args.hidden * (1 if args.fp8 else 2) + (args.hidden // 128 * 4 if args.fp8 else 0)

    info = dict(
        rank=rank, host=socket.gethostname(), local_rank=local_rank, group_id=group_id,
        per_node_tokens=be["per_node_tokens"], dispatch_us=dispatch_us, combine_us=combine_us,
    )
    everyone = [None] * world
    dist.all_gather_object(everyone, info, group=None)

    if rank == 0:
        report(args, everyone, world, be["desc"], payload_bytes)
        sys.stdout.flush()

    # Drop every CUDA tensor before UCCL tears its context down: anything freed
    # after that aborts the rank with 'invalid device context'.
    close = be.pop("close")
    be.clear()
    del combine, x, topk_idx, topk_weights
    torch.cuda.synchronize()
    dist.barrier()

    close()
    dist.destroy_process_group()
    # Hard exit: buffer.scratch outlives destroy(), and one rank aborting in
    # Python's finalizer would kill the others through --kill-on-bad-exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def report(args, everyone, world: int, desc: list, payload_bytes: int) -> None:
    everyone = sorted(everyone, key=lambda r: r["rank"])
    # Nodes in order of first appearance by rank -- the ordering UCCL's own
    # detect_group_topology() uses for node_idx, so per_node_tokens lines up.
    node_order, seen = [], set()
    for r in everyone:
        if r["host"] not in seen:
            seen.add(r["host"])
            node_order.append(r["host"])
    rpn = world // len(node_order)

    print("=" * 96)
    print(f"EP dispatch benchmark | backend {args.backend}  world {world}  "
          f"nodes {len(node_order)}  ranks/node {rpn}  EP {args.ep}  groups {world // args.ep}")
    print(f"tokens {args.num_tokens}  hidden {args.hidden}  topk {args.num_topk}  "
          f"experts {args.num_experts}  dtype {'fp8+scales' if args.fp8 else 'bf16'}  "
          f"payload {payload_bytes} B/token")
    for line in desc:
        print(line)
    print(f"iters {args.iters} (warmup {args.warmup})")
    print("=" * 96)

    rows = []
    for r in everyone:
        node_idx = node_order.index(r["host"]) % (args.ep // rpn)
        remote_tokens = sum(r["per_node_tokens"]) - r["per_node_tokens"][node_idx]
        d_med = statistics.median(r["dispatch_us"])
        rows.append(dict(
            rank=r["rank"], host=r["host"], group_id=r["group_id"],
            d_med=d_med, d_p90=pct(r["dispatch_us"], 0.90), d_max=max(r["dispatch_us"]),
            c_med=statistics.median(r["combine_us"]) if r["combine_us"] else float("nan"),
            c_p90=pct(r["combine_us"], 0.90) if r["combine_us"] else float("nan"),
            gbps=remote_tokens * payload_bytes / (d_med * 1e-6) / 1e9,
        ))

    run_med = statistics.median([w["d_med"] for w in rows])
    print("\nper rank (us)")
    print(f"{'rank':>5} {'node':>10} {'grp':>4} {'dispatch':>9} {'p90':>8} {'max':>8} "
          f"{'combine':>8} {'p90':>8} {'rdma_tx':>9}")
    for w in rows:
        flag = "  <<" if w["d_med"] > SLOW * run_med else ""
        print(f"{w['rank']:>5} {w['host']:>10} {w['group_id']:>4} {w['d_med']:>9.0f} "
              f"{w['d_p90']:>8.0f} {w['d_max']:>8.0f} {w['c_med']:>8.0f} {w['c_p90']:>8.0f} "
              f"{w['gbps']:>8.1f}G{flag}")

    print(f"\nper node (median over its {rpn} ranks, us) -- worst dispatch first, "
          f"'<<' = > {SLOW}x the run median")
    print(f"{'node':>10} {'grp':>4} {'dispatch':>9} {'p90':>8} {'combine':>8} {'p90':>8} {'rdma_tx':>9}")
    node_rows = []
    for h in node_order:
        mine = [w for w in rows if w["host"] == h]
        node_rows.append(dict(
            host=h, group_id=mine[0]["group_id"],
            d_med=statistics.median([w["d_med"] for w in mine]),
            d_p90=statistics.median([w["d_p90"] for w in mine]),
            c_med=statistics.median([w["c_med"] for w in mine]),
            c_p90=statistics.median([w["c_p90"] for w in mine]),
            gbps=statistics.median([w["gbps"] for w in mine]),
        ))
    for n in sorted(node_rows, key=lambda n: -n["d_med"]):
        flag = "  <<" if n["d_med"] > SLOW * run_med else ""
        print(f"{n['host']:>10} {n['group_id']:>4} {n['d_med']:>9.0f} {n['d_p90']:>8.0f} "
              f"{n['c_med']:>8.0f} {n['c_p90']:>8.0f} {n['gbps']:>8.1f}G{flag}")

    print("\nper EP group")
    for gid in range(world // args.ep):
        mine = [w for w in rows if w["group_id"] == gid]
        nodes = [h for h in node_order if any(w["host"] == h for w in mine)]
        meds = [w["d_med"] for w in mine]
        print(f"  grp {gid}: dispatch med {statistics.median(meds):.0f} us  "
              f"min {min(meds):.0f}  max {max(meds):.0f}  spread {max(meds) / min(meds):.2f}x")
        print(f"         nodes {','.join(nodes)}")

    # One greppable line so a sweep over candidate node sets can be diffed
    # without re-parsing the tables.
    worst = max(node_rows, key=lambda n: n["d_med"])
    print(f"\nEPBENCH backend={args.backend} dispatch_med_us={run_med:.0f} "
          f"dispatch_p90_us={pct([w['d_p90'] for w in rows], 0.90):.0f} "
          f"combine_med_us={statistics.median([w['c_med'] for w in rows]):.0f} "
          f"worst_node={worst['host']} worst_med_us={worst['d_med']:.0f} "
          f"worst_ratio={worst['d_med'] / run_med:.2f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(config=vars(args), world=world, ranks_per_node=rpn,
                           nodes=node_order, ranks=everyone), f)
        print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
