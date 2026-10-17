# Where the time goes: per-stage breakdown

From `bench/bench_breakdown.py`. A100-SXM4-40GB (Colab), torch 2.11.0+cu130,
4096 tokens, top-2. Same pipelines as `bench_moe.py` (imported, not copied),
each checked against an fp32 reference first and run in its own process.
Stage time = GPU time between CUDA events placed after each stage, so it
includes GPU idle time after a host sync. Medians of 20. Recorded 2026-10-17.

## Fine-grained (D = 1024, F = 4096, 64 experts), uniform routing, ms

| stage | ours | MegaBlocks-style (stk) | grouped GEMM | ScatterMoE |
|---|---:|---:|---:|---:|
| routing / sort | 0.04 | 0.13 (+ host sync) | 0.23 (sort + gather) | 0.49 (+ host sync) |
| padded copy of x | (fused into SDD) | 0.21 | - | (fused) |
| layout metadata | (part of routing) | 0.63 * | - | - |
| first matmul | 0.75 (SDD + GELU fused) | 0.73 | 1.16 | 1.35 (gather fused) |
| GELU | (fused) | 0.16 | 0.10 | 0.10 |
| second matmul | 0.68 | 0.66 | 1.50 | 1.50 (scatter + weights fused) |
| combine | 0.21 | 0.22 | 0.27 | (fused) |
| **total** | **1.67** | 2.74 | 3.24 | 3.44 |

Zipf routing gives the same picture (ours 1.72, stk 2.92, grouped GEMM 3.38,
ScatterMoE 3.49).

## Mixtral-like (D = 4096, F = 14336, 8 experts), uniform routing, ms

| stage | ours | MegaBlocks-style (stk) | grouped GEMM | ScatterMoE |
|---|---:|---:|---:|---:|
| routing / sort | 0.06 | 0.16 (+ sync) | 0.30 | 0.52 (+ sync) |
| padded copy of x | (fused) | 0.35 | - | (fused) |
| layout metadata | - | 0.63 * | - | - |
| first matmul | 5.99 | 6.05 | 4.25 | 9.68 |
| GELU | (fused) | 0.38 | 0.34 | 0.34 |
| second matmul | 5.72 | 4.79 | 4.00 | 8.50 |
| combine | 0.74 | 0.73 | 0.90 | (fused) |
| **total** | 12.51 | 13.11 | **9.80** | 19.01 |

## What it shows

1. **Fine-grained, vs MegaBlocks-style:** the matmuls are a tie (SDD 0.75 vs
   0.73, DSD 0.68 vs 0.66 ms). The whole gap is outside the matmuls: no
   padded copy of x, GELU fused into SDD, and a GPU-side layout with no
   host sync.
2. **\* Caveat on that gap:** stk's "layout metadata" (0.63 ms) is our PyTorch
   rebuild of MegaBlocks' block-sparse layout. MegaBlocks builds it with its
   own CUDA kernels, which are likely faster. Even with that stage removed,
   ours is still 1.3x faster (1.67 vs 2.11 ms).
3. **Fine-grained, vs grouped GEMM and ScatterMoE:** here the win IS the
   matmuls. Ours 1.43 ms for both, vs 2.66 (grouped GEMM) and 2.85
   (ScatterMoE). Many small experts suit 128-row blocks better than
   per-expert GEMMs.
4. **Mixtral-like:** grouped GEMM wins on both matmuls (4.25 + 4.00 vs
   5.99 + 5.72 ms). Our combine (0.74 ms, 6%) is a plain PyTorch gather and
   weighted sum, and could be fused into the DSD epilogue.
5. Our routing is the cheapest stage everywhere (0.04-0.06 ms) and never
   makes the CPU wait.
