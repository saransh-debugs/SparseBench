"""Routing: our CUDA kernels vs the PyTorch version (topology.build_topology).

Measures wall time per call until the result is ready on the GPU:
sync, start clock, call, sync, stop clock. That's fair to both sides. The
PyTorch version syncs inside itself, and ours doesn't, so GPU-only timing
with CUDA events would hide the cost we're trying to remove.

Writes bench/results_routing.json and prints a table.
"""

import json
import pathlib
import statistics
import time
from collections.abc import Callable

import torch

from sparsebench.routing import route_device
from sparsebench.topology import build_topology

BLOCK_SIZE = 128
FFN_DIM = 1024
WARMUP = 10
ITERS = 100


def time_us(fn: Callable[[], object]) -> float:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    rows = []
    for num_tokens in [4096, 16384, 65536]:
        for num_experts in [8, 64]:
            g = torch.Generator().manual_seed(0)
            idx = torch.rand(num_tokens, num_experts, generator=g).topk(2, dim=-1).indices.cuda()
            torch_us = time_us(lambda: build_topology(idx, num_experts, FFN_DIM, BLOCK_SIZE))
            ours_us = time_us(lambda: route_device(idx, num_experts, FFN_DIM, BLOCK_SIZE))
            rows.append(
                {
                    "tokens": num_tokens,
                    "experts": num_experts,
                    "pytorch_us": round(torch_us, 1),
                    "ours_us": round(ours_us, 1),
                    "speedup": round(torch_us / ours_us, 2),
                }
            )

    print(f"GPU: {torch.cuda.get_device_name()}  (top-2, block {BLOCK_SIZE}, median of {ITERS})\n")
    print("| tokens | experts | PyTorch (us) | ours (us) | speedup |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['tokens']} | {r['experts']} | {r['pytorch_us']} | {r['ours_us']} | {r['speedup']}x |")

    out = pathlib.Path(__file__).with_name("results_routing.json")
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
