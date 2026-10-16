"""DSD kernel timing vs the cuBLAS ceiling.

Rules (see PLAN.md):
- Time the real no-sync call: worst-case buffers from route_device, so dead
  row blocks (thread blocks that only write zeros) are included in the time.
  Grouped GEMM doesn't pay that cost, so leaving it out would flatter us.
- TFLOPS counts useful FLOPs only: 2 * (tokens * k) * F * D. Padding rows
  and dead row blocks are waste, not work.
- The ceiling is one dense cuBLAS GEMM with exactly those useful FLOPs.
- CUDA events, L2 flushed before every timed call, median of ITERS.
- Times come from here, not ncu: ncu locks clocks to base speed.

Writes bench/results_dsd.json and prints two tables:
  1. every kernel version vs cuBLAS
  2. the newest version at block size 64 vs 128: smaller blocks waste less
     padding under skewed routing, but each tile does less math
  python bench/bench_dsd.py                                      # benchmark
  python bench/bench_dsd.py --profile -v 4 --shape mixtral       # a few launches for ncu (bench/ncu.sh)
"""

import argparse
import json
import math
import pathlib
from collections.abc import Callable

import torch

from sparsebench.routing import route_device
from sparsebench.spmm import dsd
from sparsebench.timing import l2_flush_buffer
from sparsebench.timing import time_ms as _time_ms
from sparsebench.workloads import routing_case

NUM_TOKENS = 4096
TOP_K = 2
BLOCK_SIZE = 128
SHAPES = {  # name: (D, F, E)
    "mixtral": (4096, 14336, 8),
    "fine_grained": (1024, 4096, 64),
}
ROUTINGS = ["random", "zipf"]
VERSIONS = [1, 2, 3, 4, 41]  # 41 = v4 with 3 stages instead of 2
BLOCK_SIZE_VERSION = 4
WARMUP = 3
ITERS = 20


def make_problem(shape: str, routing: str, bs: int = BLOCK_SIZE) -> tuple[object, torch.Tensor, torch.Tensor]:
    d_model, ffn_dim, num_experts = SHAPES[shape]
    idx = routing_case(routing, NUM_TOKENS, num_experts).cuda()
    dev = route_device(idx, num_experts, ffn_dim, bs)
    blocks = torch.randn(dev.col_indices.numel(), bs, bs, device="cuda", dtype=torch.bfloat16) / math.sqrt(ffn_dim)
    w = torch.randn(num_experts * ffn_dim, d_model, device="cuda", dtype=torch.bfloat16)
    return dev, blocks, w


def time_ms(fn: Callable[[], object], flush: torch.Tensor) -> float:
    return _time_ms(fn, flush, warmup=WARMUP, iters=ITERS)


def benchmark() -> None:
    flush = l2_flush_buffer()
    rows = []
    for shape, (d_model, ffn_dim, _) in SHAPES.items():
        useful_flops = 2 * NUM_TOKENS * TOP_K * ffn_dim * d_model
        a = torch.randn(NUM_TOKENS * TOP_K, ffn_dim, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(ffn_dim, d_model, device="cuda", dtype=torch.bfloat16)
        cublas_ms = time_ms(lambda: a @ b, flush)
        del a, b

        for routing in ROUTINGS:
            dev, blocks, w = make_problem(shape, routing)
            topo = dev.to_topology()  # sync, outside timing: only for reporting
            launched = dev.row_offsets.numel() - 1
            for version in VERSIONS:
                ms = time_ms(lambda: dsd(blocks, w, dev, version=version), flush)
                rows.append(
                    {
                        "shape": shape,
                        "routing": routing,
                        "version": version,
                        "row_blocks_real": topo.num_row_blocks,
                        "row_blocks_launched": launched,
                        "ms": round(ms, 3),
                        "tflops": round(useful_flops / ms / 1e9, 1),
                        "cublas_ms": round(cublas_ms, 3),
                        "cublas_tflops": round(useful_flops / cublas_ms / 1e9, 1),
                        "pct_of_cublas": round(100 * cublas_ms / ms, 1),
                    }
                )
            del dev, blocks, w, topo

    print(f"GPU: {torch.cuda.get_device_name()}. {NUM_TOKENS} tokens, top-{TOP_K}, block {BLOCK_SIZE}.")
    print("TFLOPS = useful FLOPs only. Time includes dead row blocks.\n")
    print("| shape | routing | ver | row blocks real / launched | ms | TFLOPS | cuBLAS TFLOPS | % of cuBLAS |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    labels = {41: "v4, 3 stages"}
    for r in rows:
        print(
            f"| {r['shape']} | {r['routing']} | {labels.get(r['version'], 'v' + str(r['version']))} | {r['row_blocks_real']} / {r['row_blocks_launched']}"
            f" | {r['ms']} | {r['tflops']} | {r['cublas_tflops']} | {r['pct_of_cublas']}% |"
        )

    # Block size under skew. Padding rows = real rows the kernel computes but
    # no token uses: each expert's group is rounded up to a multiple of bs.
    bs_rows = []
    useful_rows = NUM_TOKENS * TOP_K
    for shape, (d_model, ffn_dim, _) in SHAPES.items():
        useful_flops = 2 * useful_rows * ffn_dim * d_model
        for routing in ROUTINGS:
            for bs in (64, 128):
                dev, blocks, w = make_problem(shape, routing, bs)
                padded_rows = int(dev.padded_offsets[-1])
                ms = time_ms(lambda: dsd(blocks, w, dev, version=BLOCK_SIZE_VERSION), flush)
                bs_rows.append(
                    {
                        "shape": shape,
                        "routing": routing,
                        "block_size": bs,
                        "padding_pct": round(100 * (padded_rows - useful_rows) / padded_rows, 1),
                        "ms": round(ms, 3),
                        "tflops": round(useful_flops / ms / 1e9, 1),
                    }
                )
                del dev, blocks, w

    print(f"\nBlock size under skew (v{BLOCK_SIZE_VERSION}). Padding = computed rows no token uses.\n")
    print("| shape | routing | block size | padding | ms | TFLOPS |")
    print("|---|---|---:|---:|---:|---:|")
    for r in bs_rows:
        print(
            f"| {r['shape']} | {r['routing']} | {r['block_size']} | {r['padding_pct']}% | {r['ms']} | {r['tflops']} |"
        )

    out = pathlib.Path(__file__).with_name("results_dsd.json")
    payload = {"gpu": torch.cuda.get_device_name(), "rows": rows, "block_size_rows": bs_rows}
    out.write_text(json.dumps(payload, indent=2) + "\n")


def profile(version: int, shape: str) -> None:
    # A few launches of one problem. ncu.sh skips the first (cold) one.
    dev, blocks, w = make_problem(shape, "random")
    for _ in range(3):
        dsd(blocks, w, dev, version=version)
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="just run a few launches, for ncu")
    parser.add_argument("-v", "--version", type=int, default=1)
    parser.add_argument("--shape", choices=list(SHAPES), default="fine_grained")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    if args.profile:
        profile(args.version, args.shape)
    else:
        benchmark()


if __name__ == "__main__":
    main()
