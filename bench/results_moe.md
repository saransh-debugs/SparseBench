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

## Adding ScatterMoE (sixth run)

ScatterMoE's own `MLP` module (commit 47b5e15), our weights loaded in its
native layout. Passed the correctness check. Same session as the rows below.

| pipeline | host sync | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---|---:|---:|---:|---:|
| **ours** | no | 12.10 | 14.63 † | **1.86** | **1.99** |
| torch._grouped_mm pipeline | no | **9.87** | **10.91** | 3.16 | 3.31 |
| stk pipeline (MegaBlocks-style) | yes | 14.59 | 12.73 | 2.62 | 2.81 |
| ScatterMoE | yes | 19.00 | 17.91 | 3.40 | 3.46 |
| per-expert loop | yes | 10.57 | 12.71 | 15.43 | 15.72 |

- Fine-grained: ours is 1.7-1.8x faster than ScatterMoE, 1.4x faster than the
  MegaBlocks-style pipeline, 1.7x faster than grouped GEMM.
- Mixtral-like: ours is 1.2-1.6x faster than ScatterMoE; grouped GEMM still wins.
- Here the MegaBlocks-style pipeline beats ScatterMoE too, the opposite of
  ScatterMoE's paper. That paper measures training (forward + backward) and
  memory; this is a forward pass only, on A100, with current Triton.
- † Outlier: the same case measured 11.88 ms in the fifth run, with the same
  code. A single-run spike like this is why the tables are compared row by
  row within one run.
