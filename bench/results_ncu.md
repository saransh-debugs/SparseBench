# ncu counters, by kernel version

One DSD launch, profiled with `bench/ncu.sh`. Counters only: ncu locks clocks
to base speed, so times come from `bench/bench_dsd.py`, not from here.

Problem: fine-grained shape (D = 1024, F = 4096, E = 64), 4096 tokens, top-2,
uniform routing, block size 128, worst-case buffers (127 row blocks launched,
92 real). GPU: A100-SXM4-40GB (Colab). Recorded 2026-10-15.

| Counter | v1 | v2 | v3 | What changed |
|---|---:|---:|---:|---|
| Tensor Core busy (% of peak) | 14.0 | 21.4 | 43.2 | |
| DRAM throughput (% of peak) | 26.9 | 42.3 | 78.2 | v3: memory bandwidth becomes the limit |
| Shared-memory load bank conflicts | 144.7M | 144.7M | 9.5k | v3: swizzle + ldmatrix |
| Shared-memory store bank conflicts | 74k | 0 | 0 | v2: cp.async writes 16 B at a time |
| Stall: long_scoreboard (waiting on global memory) | 10.7 | 0.33 | 1.90 | v2: cp.async pipeline |
| Stall: mio_throttle (shared-memory queue full) | 9.90 | 9.88 | 0.20 | v3: no bank conflicts |
| Stall: barrier | 1.74 | 1.65 | 0.72 | |
| Stall: short_scoreboard | 0.73 | 1.17 | 0.86 | |
| Stall: math_pipe_throttle (Tensor Cores busy) | 0.24 | 0.33 | 1.34 | |

Stalls are warps waiting per issued instruction
(`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`).

## v3 vs v4 (second run, both shapes)

Same setup. Mixtral-like = D = 4096, F = 14336, E = 8 (71 row blocks
launched, 69 real). v3 uses 3 stages, v4 uses 2.

| Counter | fine v3 | fine v4 | Mixtral v3 | Mixtral v4 |
|---|---:|---:|---:|---:|
| DRAM bytes read | 1.24 GB | 0.66 GB | 6.35 GB | 3.37 GB |
| DRAM throughput (% of peak) | 78.0 | 52.9 | 48.8 | 27.2 |
| L2 hit rate (%) | 35.6 | 55.3 | 57.3 | 71.9 |
| Tensor Core busy (% of peak) | 42.7 | 50.9 | 52.1 | 54.1 |
| Registers per thread | 122 | 126 | 122 | 126 |
| Blocks per SM, register limit | 2 | 2 | 2 | 2 |
| Blocks per SM, shared-memory limit | 2 | 3 | 2 | 3 |
| Warps active (% of peak) | 23.7 | 23.2 | 24.5 | 24.5 |
| Stall: long_scoreboard | 1.93 | 0.36 | 0.05 | 0.11 |
| Stall: barrier | 0.72 | 0.93 | 1.13 | 1.12 |
| Stall: math_pipe_throttle | 1.33 | 1.33 | 1.40 | 1.36 |
| Stall: short_scoreboard | 0.87 | 0.83 | 0.73 | 0.85 |
| Shared-memory load bank conflicts | 9.7k | 3.7k | 17k | 15k |

What it shows:
- The row-block-first order halves DRAM bytes read on both shapes.
- Registers cap every version at 2 blocks (16 warps) per SM, which is the
  ~24% warps active. Shared memory is never the limit, so the stage count
  doesn't change occupancy. That's why 2 vs 3 stages barely matters.
- Mixtral v4 uses only 27% of DRAM bandwidth: memory is no longer its limit.
  What's left is the GEMM inner loop itself (low occupancy, small per-warp tile).

## v4 vs v5 (third run, both shapes)

v5: 128 x 256 thread-block tile, K step 64, 64 x 64 warp tiles, 3 stages in
144 KB dynamic shared memory.

| Counter | fine v4 | fine v5 | Mixtral v4 | Mixtral v5 |
|---|---:|---:|---:|---:|
| Tensor Core busy (% of peak) | 50.8 | 62.0 | 54.1 | 65.5 |
| DRAM bytes read | 0.66 GB | 0.65 GB | 3.37 GB | 2.37 GB |
| L2 hit rate (%) | 55.6 | 47.9 | 71.9 | 72.9 |
| Registers per thread | 126 | 240 | 126 | 240 |
| Blocks per SM (registers / shared memory) | 2 / 3 | 1 / 1 | 2 / 3 | 1 / 1 |
| Warps active (% of peak) | 23.2 | 12.4 | 24.5 | 12.5 |
| Shared-memory load bank conflicts | 3.4k | 0 | 16.5k | 0 |
| Stall: math_pipe_throttle | 1.33 | 1.84 | 1.36 | 1.86 |
| Stall: barrier | 0.93 | 0.81 | 1.12 | 0.83 |
| Stall: short_scoreboard | 0.83 | 0.20 | 0.85 | 0.20 |

What it shows:
- Tensor Cores go from about half busy to about two-thirds busy. The top stall
  is now math_pipe_throttle: warps waiting because the Tensor Cores are busy.
- Half the warps per SM (1 fat block instead of 2), as planned.
- 240 registers per thread, under the 255 limit (spill check: notebook's
  cuobjdump cell).

## SDD (fourth run)

SDD = the first MoE matmul, on the shared v5 inner loop, with the token gather
and GELU fused. Spill check (`cuobjdump -res-usage`): DSD v5 and SDD both use
242 registers, STACK 0, LOCAL 0, so no register spills.

| Counter | SDD fine-grained | SDD Mixtral |
|---|---:|---:|
| Tensor Core busy (% of peak) | 51.0 | 63.3 |
| DRAM bytes read | 0.58 GB | 6.69 GB |
| DRAM throughput (% of peak) | 52.5 | 64.3 |
| L2 hit rate (%) | 53.2 | 48.6 |
| Shared-memory load bank conflicts | 0 | 0 |
| Stall: math_pipe_throttle | 1.09 | 1.47 |
| Stall: long_scoreboard | 0.55 | 0.33 |
| Stall: barrier | 0.52 | 0.77 |

Mixtral SDD reads 6.7 GB: each 128-row block streams its expert's whole
4096 x 14336 W1 slice, and with 1 thread block per SM only a few row blocks of
the same expert run at once to share it through L2.
