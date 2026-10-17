"""Token rounding (SonicMoE) vs plain top-k: speed, and how much routing changes.

For each problem, the same router scores produce two routings:
  top-k            plain top-2 (weights renormalized over the 2 picks)
  token rounding   each expert's count rounded to the nearest multiple of 128
                   (sparsebench/token_rounding.py)
and both run through:
  ours             moe_forward_assigned: route_device -> SDD -> DSD v5 -> combine
  grouped GEMM     torch._grouped_mm twice, same flat routing
Every result is checked against an fp32 reference on the same routing first.

Problems: fine-grained (64 experts) at 4096 tokens (1 tile per expert on
average, below the paper's "at least 2 tiles" quality condition) and 16384
tokens (4 tiles), and Mixtral-like at 4096 tokens. Uniform and Zipf scores.

We can't measure model quality without training. The routing-change columns
are the proxy; the paper reports quality (SonicMoE, Table 2).

Runs in one process: only our kernels and torch's run here (bench_moe.py
isolates third-party baselines because one crashed the CUDA context).
CUDA events, L2 flushed, median of 20.

  python bench/bench_token_rounding.py
"""

import json
import pathlib

import torch
import torch.nn.functional as F

from sparsebench.moe import moe_forward_assigned
from sparsebench.timing import l2_flush_buffer, time_ms
from sparsebench.token_rounding import Assignments, routing_change, token_rounding, top_k_assignments
from sparsebench.workloads import router_scores

TOP_K = 2
TILE = 128
PROBLEMS = [  # (shape, tokens, D, F, E)
    ("fine_grained", 4096, 1024, 4096, 64),
    ("fine_grained", 16384, 1024, 4096, 64),
    ("mixtral", 4096, 4096, 14336, 8),
]
ROUTINGS = ["uniform", "zipf"]
TOL = 3e-2


def grouped_mm_assigned(x, a: Assignments, w1, w2) -> torch.Tensor:
    num_experts = w1.shape[0]
    order = torch.sort(a.expert, stable=True).indices
    counts = torch.zeros(num_experts, dtype=torch.int32, device=x.device).scatter_add_(
        0, a.expert, torch.ones_like(a.expert, dtype=torch.int32)
    )
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    token = a.token[order]
    h = F.gelu(torch._grouped_mm(x[token], w1, offs=offs))
    y = torch._grouped_mm(h, w2, offs=offs).float() * a.weight[order, None]
    return torch.zeros(x.shape[0], x.shape[1], device=x.device).index_add_(0, token, y).to(x.dtype)


def reference(x, a: Assignments, w1, w2) -> torch.Tensor:
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device)
    for e in range(w1.shape[0]):
        m = a.expert == e
        if m.any():
            y = F.gelu(x[a.token[m]].float() @ w1[e].float()) @ w2[e].float()
            out.index_add_(0, a.token[m], y * a.weight[m, None])
    return out


def checked_time(fn, ref: torch.Tensor, flush: torch.Tensor) -> float | str:
    err = (fn().float() - ref).abs().max().item()
    if err > TOL * (1 + ref.abs().max().item()):
        return f"wrong (max error {err:.3g})"
    return round(time_ms(fn, flush), 3)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    flush = l2_flush_buffer()
    has_gmm = hasattr(torch, "_grouped_mm")
    rows = []
    for shape, num_tokens, d_model, ffn_dim, num_experts in PROBLEMS:
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(num_tokens, d_model, generator=g, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(num_experts, d_model, ffn_dim, generator=g, device="cuda", dtype=torch.bfloat16) / d_model**0.5
        w2 = torch.randn(num_experts, ffn_dim, d_model, generator=g, device="cuda", dtype=torch.bfloat16) / ffn_dim**0.5
        for routing in ROUTINGS:
            scores = router_scores(routing, num_tokens, num_experts).cuda()
            plain = top_k_assignments(scores, TOP_K)
            rounded = token_rounding(scores, TOP_K, TILE)
            row: dict = {"shape": shape, "tokens": num_tokens, "routing": routing,
                         "tiles_per_expert": num_tokens * TOP_K / num_experts / TILE}  # fmt: skip
            row |= routing_change(scores, TOP_K, rounded, TILE)
            row["rounding_ms"] = round(time_ms(lambda: token_rounding(scores, TOP_K, TILE), flush), 3)
            for label, a in (("topk", plain), ("tr", rounded)):
                ref = reference(x, a, w1, w2)
                row[f"ours_{label}_ms"] = checked_time(
                    lambda a=a: moe_forward_assigned(x, a.token, a.expert, a.weight, w1, w2), ref, flush
                )
                row[f"gmm_{label}_ms"] = (
                    checked_time(lambda a=a: grouped_mm_assigned(x, a, w1, w2), ref, flush) if has_gmm else "n/a"
                )
                del ref
            rows.append(row)
        del x, w1, w2
        torch.cuda.empty_cache()

    def speedup(before: object, after: object) -> str:
        return f"{before / after:.2f}x" if isinstance(before, float) and isinstance(after, float) else "-"

    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}. top-{TOP_K}, tile {TILE}.\n")
    print("### Speed: whole layer (ms), plain top-k vs token rounding\n")
    print("| shape | tokens | routing | tiles/expert | ours top-k | ours TR | ours gain | grouped GEMM top-k | grouped GEMM TR | GEMM gain | rounding itself |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['shape']} | {r['tokens']} | {r['routing']} | {r['tiles_per_expert']:.0f} | {r['ours_topk_ms']} | {r['ours_tr_ms']} "
            f"| {speedup(r['ours_topk_ms'], r['ours_tr_ms'])} | {r['gmm_topk_ms']} | {r['gmm_tr_ms']} "
            f"| {speedup(r['gmm_topk_ms'], r['gmm_tr_ms'])} | {r['rounding_ms']} |"
        )
    print("\n### How much routing changed (% of top-k picks, except where noted)\n")
    print("| shape | tokens | routing | top-k padding (% of rows) | picks dropped | assignments added | tokens with 0 experts (% of tokens) | tokens with > k (% of tokens) | top-k score kept |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['shape']} | {r['tokens']} | {r['routing']} | {r['topk_padding_pct']:.1f} | {r['topk_picks_dropped_pct']:.1f} "
            f"| {r['assignments_added_pct']:.1f} | {r['tokens_with_0_experts_pct']:.2f} | {r['tokens_with_more_than_k_pct']:.1f} "
            f"| {r['topk_score_mass_kept_pct']:.1f} |"
        )
    out = pathlib.Path(__file__).with_name("results_token_rounding.json")
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
