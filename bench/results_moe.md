# Whole MoE layer, end to end

From `bench/bench_moe.py`: routing + both matmuls + GELU + combine, everything
from `expert_idx` to the final output inside the timer, host syncs included.
A100-SXM4-40GB (Colab), torch 2.11.0+cu130, 4096 tokens, top-2. Each pipeline
runs in its own process and is checked against an fp32 reference before
timing. stk uses its fastest correct index widths. Recorded 2026-10-16.

## Time in ms (lower is better)

| pipeline | host sync | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---|---:|---:|---:|---:|
| **ours (route + SDD + DSD v5)** | no | 12.06 | 11.88 | **1.85** | **1.99** |
| torch._grouped_mm pipeline | no | **9.75** | **10.38** | 3.17 | 3.33 |
| stk pipeline (MegaBlocks-style) | yes | 13.02 | 12.69 | 2.60 | 2.81 |
| per-expert loop | yes | 12.75 | 10.40 | 15.01 | 16.36 |

## What it shows

- **Fine-grained (64 experts):** ours is 1.4x faster than the MegaBlocks-style
  pipeline and 1.7x faster than grouped GEMM, with no host sync. Our DSD alone
  only ties stk's (`results_baselines.md`), so the end-to-end win comes from
  the rest of the layer: SDD with the gather and GELU fused, routing on the
  GPU, and no CPU round trip.
- **Mixtral-like (8 experts):** grouped GEMM wins by 1.15-1.25x. Ours still
  beats the MegaBlocks-style pipeline by ~1.07x.

## Run-to-run noise

Absolute times moved up to ~20% between Colab sessions (fine-grained ours:
1.49 ms in the fourth run, 1.85 ms here, same code), and the baselines moved
with them. Ratios inside one run are steadier: compare rows within a table,
not across runs.
