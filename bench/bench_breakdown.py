"""Where does the time go? Per-stage times of the whole MoE layer.

Same pipelines and data as bench_moe.py (imported from it, not copied): each
pipeline calls mark(stage) right after launching a stage, and we record a
CUDA event there. Stage time = GPU time between consecutive events, so it
also counts any time the GPU sat idle waiting for the CPU, e.g. after a host
sync. That idle time is part of the real cost of a sync.

Each pipeline is checked against the fp32 reference first, runs in its own
process (as in bench_moe.py), L2 flushed before every run, median of 20.

  python bench/bench_breakdown.py
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import bench_moe as bm  # same directory
import torch

from sparsebench.moe import moe_forward
from sparsebench.timing import l2_flush_buffer

PIPELINES = ["ours", "torch._grouped_mm", "stk (MegaBlocks-style)", "ScatterMoE"]
WARMUP = 3
ITERS = 20


def marked_pipeline(name: str, w1: torch.Tensor, w2: torch.Tensor):
    """fn(x, idx, weight, w1, w2, mark) for each pipeline, or None if unavailable."""
    if name == "ours":
        return lambda x, idx, wt, w1, w2, mark: moe_forward(x, idx, wt, w1, w2, mark=mark)
    if name == "torch._grouped_mm":
        return bm.grouped_mm_pipeline if hasattr(torch, "_grouped_mm") else None
    if name == "stk (MegaBlocks-style)":
        try:
            import stk  # noqa: F401
        except ImportError:
            return None
        num_experts, d_model, ffn_dim = w1.shape
        w1_cat = w1.permute(1, 0, 2).reshape(d_model, num_experts * ffn_dim).contiguous()
        w2_cat = w2.reshape(num_experts * ffn_dim, d_model)
        return lambda x, idx, wt, w1, w2, mark: bm.stk_pipeline(x, idx, wt, w1, w2, w1_cat, w2_cat, mark=mark)
    if name == "ScatterMoE":
        try:
            mlp = bm.make_scattermoe_mlp(w1, w2)
        except ImportError:
            return None
        return lambda x, idx, wt, w1, w2, mark: bm.scattermoe_pipeline(x, idx, wt, mlp, mark=mark)
    raise ValueError(name)


def run_one(shape: str, routing: str, name: str) -> dict:
    x, idx, weight, w1, w2 = bm.make_layer(shape, routing)
    fn = marked_pipeline(name, w1, w2)
    if fn is None:
        return {"pipeline": name, "skipped": "not installed / not supported"}
    ref = bm.reference_output(x, idx, weight, w1, w2)
    err = (fn(x, idx, weight, w1, w2, lambda _: None).float() - ref).abs().max().item()
    if err > bm.TOL * (1 + ref.abs().max().item()):
        return {"pipeline": name, "skipped": f"wrong answer (max error {err:.3g})"}

    flush = l2_flush_buffer()
    for _ in range(WARMUP):
        fn(x, idx, weight, w1, w2, lambda _: None)
    stages: dict[str, list[float]] = {}
    totals = []
    for _ in range(ITERS):
        flush.zero_()
        events: list[tuple[str, torch.cuda.Event]] = []

        def mark(stage: str) -> None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.append((stage, e))

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x, idx, weight, w1, w2, mark)
        torch.cuda.synchronize()
        prev = start
        for stage, e in events:
            stages.setdefault(stage, []).append(prev.elapsed_time(e))
            prev = e
        totals.append(start.elapsed_time(events[-1][1]))
    return {
        "pipeline": name,
        "total_ms": round(statistics.median(totals), 3),
        "stages_ms": {s: round(statistics.median(v), 3) for s, v in stages.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=3, metavar=("SHAPE", "ROUTING", "PIPELINE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    if args.one:
        print("RESULT " + json.dumps(run_one(*args.one)))
        return

    results = []
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}. {bm.NUM_TOKENS} tokens, top-2.")
    print("Per-stage GPU time (ms), medians. Includes GPU idle time after host syncs.\n")
    for shape, (d_model, ffn_dim, num_experts) in bm.SHAPES.items():
        for routing in bm.ROUTINGS:
            rows = []
            for name in PIPELINES:
                proc = subprocess.run([sys.executable, __file__, "--one", shape, routing, name],
                                      capture_output=True, text=True, timeout=1800)  # fmt: skip
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                if proc.returncode == 0 and lines:
                    rows.append(json.loads(lines[-1][len("RESULT "):]))
                else:
                    last = (proc.stderr.strip().splitlines() or ["no output"])[-1][:150]
                    rows.append({"pipeline": name, "skipped": f"crashed (exit {proc.returncode}): {last}"})
            print(f"### {shape}, {routing} routing (D={d_model}, F={ffn_dim}, E={num_experts})\n")
            print("| pipeline | total | stages |")
            print("|---|---:|---|")
            for r in rows:
                if "skipped" in r:
                    print(f"| {r['pipeline']} | - | skipped: {r['skipped']} |")
                else:
                    parts = " · ".join(f"{s} {ms}" for s, ms in r["stages_ms"].items())
                    print(f"| {r['pipeline']} | {r['total_ms']} | {parts} |")
            print()
            results.append({"shape": shape, "routing": routing, "rows": rows})
    out = pathlib.Path(__file__).with_name("results_breakdown.json")
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
