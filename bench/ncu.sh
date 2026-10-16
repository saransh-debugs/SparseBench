#!/usr/bin/env bash
# Hardware counters for one DSD kernel launch.
#
# Counters only. ncu locks the GPU clock to base speed, so its timings won't
# match bench_dsd.py. Times come from bench_dsd.py.
#
#   bash bench/ncu.sh [version] [shape]     e.g. bash bench/ncu.sh 5 mixtral
#   version: a DSD version (1-5, or a hidden variant like 41), or "sdd".
#   shape: fine_grained (default) or mixtral.
set -euo pipefail
VERSION="${1:-1}"
SHAPE="${2:-fine_grained}"
if [[ "$VERSION" == "sdd" ]]; then
  KERNEL="sdd_kernel"
  CMD=(python bench/bench_moe.py --profile --shape "$SHAPE")
else
  KERNEL="dsd_v${VERSION:0:1}_kernel"   # 41 runs the v4 kernel
  CMD=(python bench/bench_dsd.py --profile --version "$VERSION" --shape "$SHAPE")
fi

# What each metric tells us:
#   hmma_cycles_active   how busy the Tensor Cores are (the goal: high)
#   bank_conflicts_*     shared-memory bank conflicts (v3 should drive these to ~0)
#   stalled_*            why warps wait, per issued instruction:
#     long_scoreboard    waiting on global memory      (v2 should cut this)
#     short_scoreboard   waiting on shared memory
#     mio_throttle       shared-memory queue is full    (bank conflicts feed this)
#     barrier            waiting at __syncthreads
#     math_pipe_throttle Tensor Cores busy: the good kind of stall
#   dram__bytes_read     bytes actually read from GPU memory (v4: tile order should cut this)
#   lts__..._hit_rate    L2 cache hit rate
#   registers / occupancy_limit_* / warps_active
#                        how many thread blocks fit per SM, what limits it, and
#                        how full the SMs actually were (explains 2 vs 3 stages)
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
  dram__bytes_read.sum
  lts__t_sector_hit_rate.pct
  launch__registers_per_thread
  launch__occupancy_limit_registers
  launch__occupancy_limit_shared_mem
  sm__warps_active.avg.pct_of_peak_sustained_active
)

ncu --kernel-name "regex:${KERNEL}" --launch-skip 1 --launch-count 1 \
    --metrics "$(IFS=,; echo "${METRICS[*]}")" \
    "${CMD[@]}"
