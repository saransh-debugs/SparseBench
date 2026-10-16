"""A whole MoE layer, end to end: routing + both matmuls + GELU + combine.

Same inputs for every pipeline: x [T, D], top-2 routing (expert_idx, weights),
w1 [E, D, F], w2 [E, F, D]. Each pipeline does everything from expert_idx to
the final [T, D] output inside the timer, including any host syncs it needs
(they are real costs here, unlike in bench_baselines.py, which times one matmul).

Pipelines:
  ours         route_device -> sdd (fused gather + GELU) -> dsd v5 -> combine. No host sync.
  grouped_mm   sort tokens by expert -> torch._grouped_mm -> GELU -> torch._grouped_mm
               -> unsort + combine. No host sync.
  stk          MegaBlocks-style: block-sparse layout on the host (sync), padded copy
               of x, stk.ops.sdd -> GELU -> stk.ops.dsd -> combine. Our routing kernels
               stand in for MegaBlocks' own.
  loop         per expert: x[tokens] @ w1[e] -> GELU -> @ w2[e]. Counts on the host (sync).

Each pipeline is checked against the fp32 reference (moe_reference) first and
only timed if it's right. CUDA events, L2 flushed, median of 20.

  python bench/bench_moe.py
  python bench/bench_moe.py --profile --shape mixtral   # a few layers, for ncu (bench/ncu.sh sdd)
"""

import argparse
import json
import pathlib
from collections.abc import Callable

import torch
import torch.nn.functional as F

from sparsebench.moe import moe_forward
from sparsebench.routing import build_topology_cuda
from sparsebench.timing import l2_flush_buffer, time_ms
from sparsebench.workloads import routing_case

NUM_TOKENS = 4096
SHAPES = {  # name: (D, F, E), same as the other benchmarks
    "mixtral": (4096, 14336, 8),
    "fine_grained": (1024, 4096, 64),
}
ROUTINGS = ["random", "zipf"]
TOL = 3e-2  # two bf16 matmuls with H rounded to bf16 in between


def make_layer(shape: str, routing: str) -> tuple[torch.Tensor, ...]:
    d_model, ffn_dim, num_experts = SHAPES[shape]
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(NUM_TOKENS, d_model, generator=g, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(num_experts, d_model, ffn_dim, generator=g, device="cuda", dtype=torch.bfloat16) / d_model**0.5
    w2 = torch.randn(num_experts, ffn_dim, d_model, generator=g, device="cuda", dtype=torch.bfloat16) / ffn_dim**0.5
    idx = routing_case(routing, NUM_TOKENS, num_experts).cuda()
    weight = torch.rand(NUM_TOKENS, 2, generator=g, device="cuda").bfloat16()
    return x, idx, weight, w1, w2


def combine(y_copies: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y_copies [T*k, D] in copy order (t*k + j) -> weighted sum per token."""
    num_tokens, k = weight.shape
    return (y_copies.view(num_tokens, k, -1).float() * weight.float()[..., None]).sum(1).to(y_copies.dtype)


def grouped_mm_pipeline(x, idx, weight, w1, w2) -> torch.Tensor:
    num_experts, k = w1.shape[0], idx.shape[1]
    flat = idx.reshape(-1)
    order = torch.sort(flat, stable=True).indices
    # Counts without torch.bincount, which syncs on CUDA to find its length.
    counts = torch.zeros(num_experts, dtype=torch.int32, device=x.device).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int32)
    )
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    h = F.gelu(torch._grouped_mm(x[order // k], w1, offs=offs))
    y = torch._grouped_mm(h, w2, offs=offs)
    y_copies = torch.empty_like(y)
    y_copies[order] = y
    return combine(y_copies, weight)


def stk_pipeline(x, idx, weight, w1, w2, w1_cat, w2_cat) -> torch.Tensor:
    import stk

    num_experts, d_model, ffn_dim = w1.shape
    topo = build_topology_cuda(idx, num_experts, ffn_dim, 128)  # syncs: stk needs sizes on the host
    row_token = topo.row_token.long()
    xg = x[row_token.clamp(min=0)] * (row_token >= 0)[:, None].to(x.dtype)  # padded, grouped copy of x
    counts = (topo.row_offsets[1:] - topo.row_offsets[:-1]).long()
    row_indices = torch.repeat_interleave(torch.arange(topo.num_row_blocks, device=x.device), counts)
    cols = topo.col_indices.long()
    gather = torch.sort(cols, stable=True).indices
    num_block_cols = num_experts * ffn_dim // 128
    offsets_t = torch.zeros(num_block_cols + 1, dtype=torch.int32, device=x.device)
    offsets_t[1:] = torch.cumsum(torch.bincount(cols, minlength=num_block_cols), 0)
    idx_t = torch.int16 if max(topo.num_row_blocks, num_block_cols) < 2**15 else torch.int32
    layout = stk.Matrix(
        (topo.num_rows, num_experts * ffn_dim),
        torch.empty(topo.nnz_blocks, 128, 128, dtype=x.dtype, device=x.device),
        row_indices.to(idx_t), cols.to(idx_t), topo.row_offsets.to(torch.int32),
        row_indices[gather].to(idx_t), offsets_t, gather.to(torch.int32),
    )  # fmt: skip
    h = stk.ops.sdd(xg, w1_cat, layout)
    h = stk.Matrix(h.size(), F.gelu(h.data), h.row_indices, h.column_indices, h.offsets,
                   h.column_indices_t, h.offsets_t, h.block_offsets_t)  # fmt: skip
    y = stk.ops.dsd(h, w2_cat)
    return combine(y[topo.copy_row.long()], weight)


def loop_pipeline(x, idx, weight, w1, w2) -> torch.Tensor:
    k = idx.shape[1]
    flat = idx.reshape(-1)
    y_copies = torch.empty(flat.numel(), x.shape[1], device=x.device, dtype=x.dtype)
    for e, n in enumerate(torch.bincount(flat, minlength=w1.shape[0]).tolist()):  # host sync
        if n:
            copies = torch.nonzero(flat == e).squeeze(1)
            y_copies[copies] = F.gelu(x[copies // k] @ w1[e]) @ w2[e]
    return combine(y_copies, weight)


def pipelines(w1: torch.Tensor, w2: torch.Tensor) -> list[tuple[str, bool, Callable, str | None]]:
    """(name, host sync?, fn(x, idx, weight, w1, w2), reason it can't run or None)."""
    num_experts, d_model, ffn_dim = w1.shape
    out = [("ours (route + sdd + dsd v5)", False, moe_forward, None)]
    out.append(("torch._grouped_mm pipeline", False, grouped_mm_pipeline,
                None if hasattr(torch, "_grouped_mm") else "no torch._grouped_mm"))  # fmt: skip
    try:
        import stk  # noqa: F401

        # Concatenated weight copies, built once outside the timer (MegaBlocks stores weights this way).
        w1_cat = w1.permute(1, 0, 2).reshape(d_model, num_experts * ffn_dim).contiguous()
        w2_cat = w2.reshape(num_experts * ffn_dim, d_model)
        out.append(("stk pipeline (MegaBlocks-style)", True,
                    lambda *a: stk_pipeline(*a, w1_cat, w2_cat), None))  # fmt: skip
    except ImportError as e:
        out.append(("stk pipeline (MegaBlocks-style)", True, None, f"stk not installed ({e})"))
    out.append(("per-expert loop", True, loop_pipeline, None))
    return out


def benchmark() -> None:
    flush = l2_flush_buffer()
    results = []
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}. {NUM_TOKENS} tokens, top-2.")
    print("Whole layer, routing included. 'vs ours' > 1 means slower than ours.\n")
    for shape, (d_model, ffn_dim, num_experts) in SHAPES.items():
        for routing in ROUTINGS:
            x, idx, weight, w1, w2 = make_layer(shape, routing)
            # fp32 reference (same math as reference.moe_reference), one expert at
            # a time so fp32 copies of the weights never all exist at once.
            ref = torch.zeros(NUM_TOKENS, d_model, device="cuda")
            for e in range(num_experts):
                token, slot = torch.nonzero(idx == e, as_tuple=True)
                if token.numel():
                    y = F.gelu(x[token].float() @ w1[e].float()) @ w2[e].float()
                    ref.index_add_(0, token, y * weight[token, slot, None].float())
            scale = 1 + ref.abs().max().item()

            rows = []
            for name, sync, fn, reason in pipelines(w1, w2):
                row: dict = {"method": name, "host_sync": sync}
                if reason is None:
                    try:
                        err = (fn(x, idx, weight, w1, w2).float() - ref).abs().max().item()
                        if err > TOL * scale:
                            reason = f"wrong answer (max error {err:.3g}), not timed"
                        else:
                            ms = time_ms(lambda: fn(x, idx, weight, w1, w2), flush)
                            flops = 2 * 2 * NUM_TOKENS * 2 * d_model * ffn_dim  # 2 matmuls, k = 2 copies
                            row |= {"ms": round(ms, 3), "tflops": round(flops / ms / 1e9, 1)}
                    except Exception as e:  # noqa: BLE001  report any failure, keep going
                        reason = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
                if reason is not None:
                    row["skipped"] = reason
                rows.append(row)
                torch.cuda.empty_cache()

            ours_ms = rows[0].get("ms")
            print(f"### {shape}, {routing} routing (D={d_model}, F={ffn_dim}, E={num_experts})\n")
            print("| pipeline | ms | TFLOPS | vs ours | host sync |")
            print("|---|---:|---:|---:|---|")
            for r in rows:
                sync = "yes" if r["host_sync"] else "no"
                if "ms" in r:
                    vs = f"{r['ms'] / ours_ms:.2f}x" if ours_ms else "-"
                    print(f"| {r['method']} | {r['ms']} | {r['tflops']} | {vs} | {sync} |")
                else:
                    print(f"| {r['method']} | - | - | - | {sync} | skipped: {r['skipped']} |")
            print()
            results.append({"shape": shape, "routing": routing, "rows": rows})
            del x, idx, weight, w1, w2, ref
            torch.cuda.empty_cache()

    out = pathlib.Path(__file__).with_name("results_moe.json")
    payload = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}
    out.write_text(json.dumps(payload, indent=2) + "\n")


def profile(shape: str) -> None:
    # A few whole layers. ncu.sh skips the first (cold) SDD launch.
    x, idx, weight, w1, w2 = make_layer(shape, "random")
    for _ in range(3):
        moe_forward(x, idx, weight, w1, w2)
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="just run a few layers, for ncu")
    parser.add_argument("--shape", choices=list(SHAPES), default="fine_grained")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    if args.profile:
        profile(args.shape)
    else:
        benchmark()


if __name__ == "__main__":
    main()
