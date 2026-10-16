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

## Update: v5 (third run)

Same setup. Times in ms.

| method | host sync | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---|---:|---:|---:|---:|
| **ours (v5)** | no | 6.44 | 5.53 | 0.637 | 0.684 |
| ours (v4) | no | 6.98 | 6.53 | 0.703 | 0.763 |
| torch._grouped_mm | no | 4.13 | 4.12 | 1.206 | 1.284 |
| grouped_gemm (CUTLASS) | yes | 4.10 | 4.11 | 1.173 | 1.218 |
| stk (MegaBlocks Triton DSD) | yes | 4.96 | 4.97 | 0.629 | 0.704 |
| per-expert cuBLAS loop | yes | 4.08 | 4.09 | 1.899 | 1.981 |
| padded bmm | yes | 3.96 | 9.79 | 0.547 | 4.355 |

- **Fine-grained: v5 ties MegaBlocks' Triton DSD** (1% slower uniform, 3%
  faster Zipf) without its host sync, and is 1.8-1.9x faster than grouped GEMM.
- **Mixtral-like: still behind.** 1.1-1.3x behind stk, 1.35-1.55x behind
  grouped GEMM. With 8 big experts there is nothing sparse to exploit, and
  tuned dense GEMM tiles win.
- Run-to-run noise: the same v5 Mixtral-uniform case measured 5.67 ms in
  bench_dsd.py and 6.44 ms here. Differences under ~10% between scripts are
  not meaningful; compare rows within one table.

## Final fair run (fifth run): stk on its fastest correct index widths

int16 block indices for stk's DSD (MegaBlocks' own choice, checked correct
every run), int32 only where stk needs it (its SDD). Times in ms.

| method | host sync | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---|---:|---:|---:|---:|
| **ours (v5)** | no | 6.41 | 5.52 | 0.630 | 0.679 |
| torch._grouped_mm | no | 4.04 | 4.05 | 1.192 | 1.244 |
| grouped_gemm (CUTLASS) | yes | 3.95 | 4.05 | 1.180 | 1.211 |
| stk (MegaBlocks Triton DSD) | yes | 4.89 | 5.03 | 0.649 | 0.696 |
| per-expert cuBLAS loop | yes | 4.07 | 4.05 | 1.904 | 1.959 |
| padded bmm | yes | 4.00 | 9.73 | 0.546 | 4.398 |

Fine-grained: ours and stk within 3% of each other (ours slightly ahead), both
~1.8x faster than grouped GEMM. Mixtral-like: grouped GEMM and stk ahead.

## Adding ScatterMoE (sixth run)

ScatterMoE (Tan et al. 2024), commit 47b5e15, its `parallel_linear` on
grouped rows. Passed the correctness check. Times in ms.

| method | Mixtral uniform | Mixtral Zipf | fine-grained uniform | fine-grained Zipf |
|---|---:|---:|---:|---:|
| **ours (v5)** | 5.93 | 5.52 | 0.627 | 0.676 |
| stk (MegaBlocks Triton DSD) | 4.95 | 5.04 | 0.628 | 0.700 |
| ScatterMoE | 8.43 | 7.10 | 1.554 | 1.445 |

On this one matmul, forward only, on A100: ScatterMoE is 1.3-1.4x slower
than ours on Mixtral-like shapes and 2.1-2.5x slower on fine-grained. Its
kernel ships a single tile config (128 x 128 x 32, 4 warps) for every shape.
ScatterMoE's paper reports wins over MegaBlocks for *training* (forward +
backward, and memory), which this benchmark doesn't measure.
