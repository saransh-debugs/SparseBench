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

v4 numbers (and the new DRAM-bytes, L2-hit-rate and occupancy counters) go
here after the next run.
