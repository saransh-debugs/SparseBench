# DSD vs real MoE baselines

From `bench/bench_baselines.py`. A100-SXM4-40GB (Colab), torch 2.11.0+cu130,
4096 tokens, top-2, block size 128. Recorded 2026-10-16.

Every method was checked against the same fp32 reference before timing; none
were skipped. TFLOPS = useful FLOPs only. "Host sync" = needs routing results
on the CPU in real use (those were computed outside the timer, which favours
those methods).

## Time in ms (lower is better)

| method | host sync | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---|---:|---:|---:|---:|
| ours (v4) | no | 6.47 | 6.50 | 0.703 | 0.762 |
| torch._grouped_mm | no | 4.05 | 4.13 | 1.208 | 1.255 |
| grouped_gemm (CUTLASS, MegaBlocks) | yes | 4.05 | 4.09 | 1.176 | 1.218 |
| stk (MegaBlocks Triton DSD) | yes | 4.88 | 4.95 | 0.629 | 0.690 |
| per-expert cuBLAS loop | yes | 4.10 | 4.07 | 1.879 | 1.941 |
| padded bmm | yes | 4.04 | 9.68 | 0.546 | 4.355 |

Mixtral-like: D = 4096, F = 14336, E = 8. Fine-grained: D = 1024, F = 4096, E = 64.

## What it shows

- **Few big experts (Mixtral):** ours is the slowest, about 1.6x behind grouped
  GEMM and 1.3x behind stk. With 8 big GEMMs, cuBLAS / CUTLASS tiles win; the
  gap is our GEMM inner loop, not the sparse layout.
- **Many small experts (fine-grained):** ours is 1.7x faster than both grouped
  GEMMs and 2.7x faster than the per-expert loop. stk is still about 10% faster.
- **Padded bmm** is fast only with uniform routing; under Zipf it is 2.4x
  (Mixtral) and 5.7x (fine-grained) slower than ours.
- Among methods with **no host sync**, ours is the fastest on fine-grained MoE.
