"""GPU timing shared by every benchmark, so all numbers are measured the same way."""

import statistics
from collections.abc import Callable

import torch

L2_FLUSH_BYTES = 128 * 1024 * 1024  # > A100's 40 MB and H100's 50 MB L2


def l2_flush_buffer() -> torch.Tensor:
    return torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")


def time_ms(fn: Callable[[], object], flush: torch.Tensor, warmup: int = 3, iters: int = 20) -> float:
    """Median GPU time of fn() in ms. CUDA events, L2 flushed before every call
    so each one starts cold."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)
