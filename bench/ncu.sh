#!/usr/bin/env bash
# Hardware counters for one DSD kernel launch.
#
# Counters only. ncu locks the GPU clock to base speed, so its timings won't
# match bench_dsd.py. Times come from bench_dsd.py.
#
#   bash bench/ncu.sh [version]
set -euo pipefail
VERSION="${1:-1}"

# What each metric tells us:
#   hmma_cycles_active   how busy the Tensor Cores are (the goal: high)
#   bank_conflicts_*     shared-memory bank conflicts (v3 should drive these to ~0)
#   stalled_*            why warps wait, per issued instruction:
#     long_scoreboard    waiting on global memory      (v2 should cut this)
#     short_scoreboard   waiting on shared memory
#     mio_throttle       shared-memory queue is full    (bank conflicts feed this)
#     barrier            waiting at __syncthreads
#     math_pipe_throttle Tensor Cores busy: the good kind of stall
METRICS=(
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
)

ncu --kernel-name "regex:dsd_v${VERSION}_kernel" --launch-skip 1 --launch-count 1 \
    --metrics "$(IFS=,; echo "${METRICS[*]}")" \
    python bench/bench_dsd.py --profile --version "$VERSION"
