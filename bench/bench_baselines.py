"""Our DSD kernel vs the methods real MoE systems use, on the same data.

For every shape and routing:
  1. Build one problem (sparsebench/baselines.py): the same H and W for all.
  2. For each method: skip with a reason if it isn't installed or supported;
     otherwise check its answer against the fp32 reference, and only time it
     if the answer is right.
  3. Print a table and write bench/results_baselines.json.

Rules match bench_dsd.py: CUDA events, L2 flushed, median of 20, TFLOPS from
useful FLOPs only. Layout building isn't timed for anyone. Methods that need
routing results on the CPU get them for free (outside the timer); the
"host sync" column says what that costs them in real use.

  python bench/bench_baselines.py
"""

import json
import pathlib

import torch

from sparsebench.baselines import BASELINES, Problem, make_problem, reference
from sparsebench.timing import l2_flush_buffer, time_ms
from sparsebench.workloads import routing_case

NUM_TOKENS = 4096
BLOCK_SIZE = 128
SHAPES = {  # name: (D, F, E), same as bench_dsd.py
    "mixtral": (4096, 14336, 8),
    "fine_grained": (1024, 4096, 64),
}
ROUTINGS = ["random", "zipf"]
TOL = 1.6e-2  # bf16 output rounding; outputs are about N(0, 1)


def run_one(p: Problem, ref: torch.Tensor, flush: torch.Tensor) -> list[dict]:
    rows = []
    for b in BASELINES:
        row: dict = {"method": b.name, "host_sync": b.host_sync, "note": b.note}
        reason = b.unavailable()
        if reason is None:
            try:
                run, to_grouped = b.setup(p)
                err = (to_grouped(run()).float() - ref).abs().max().item()
                if err > TOL * (1 + ref.abs().max().item()):
                    reason = f"wrong answer (max error {err:.3g}), not timed"
                else:
                    ms = time_ms(run, flush)
                    row |= {"ms": round(ms, 3), "tflops": round(p.useful_flops / ms / 1e9, 1)}
                del run, to_grouped
            except Exception as e:  # noqa: BLE001  report any failure, keep benchmarking the rest
                reason = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
        if reason is not None:
            row["skipped"] = reason
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    flush = l2_flush_buffer()
    results = []
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}. {NUM_TOKENS} tokens, top-2.")
    print("TFLOPS = useful FLOPs only. 'vs ours' > 1 means slower than ours.\n")
    for shape, (d_model, ffn_dim, num_experts) in SHAPES.items():
        for routing in ROUTINGS:
            idx = routing_case(routing, NUM_TOKENS, num_experts).cuda()
            p = make_problem(idx, num_experts, ffn_dim, d_model, BLOCK_SIZE)
            ref = reference(p)
            rows = run_one(p, ref, flush)
            ours_ms = rows[0].get("ms")  # "vs ours" compares to the newest version (first row)
            print(f"### {shape}, {routing} routing (D={d_model}, F={ffn_dim}, E={num_experts})\n")
            print("| method | ms | TFLOPS | vs ours | host sync | note |")
            print("|---|---:|---:|---:|---|---|")
            for r in rows:
                if "ms" in r:
                    vs = f"{r['ms'] / ours_ms:.2f}x" if ours_ms else "-"
                    print(f"| {r['method']} | {r['ms']} | {r['tflops']} | {vs} | {'yes' if r['host_sync'] else 'no'} | {r['note']} |")
                else:
                    print(f"| {r['method']} | - | - | - | {'yes' if r['host_sync'] else 'no'} | skipped: {r['skipped']} |")
            print()
            results.append({"shape": shape, "routing": routing, "rows": rows})
            del p, ref
            torch.cuda.empty_cache()

    out = pathlib.Path(__file__).with_name("results_baselines.json")
    payload = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}
    out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
