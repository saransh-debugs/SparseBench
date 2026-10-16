"""The benchmark's baseline adapters must give the right answer, or every
speedup in the README is meaningless."""

import pytest
import torch

from sparsebench.baselines import BASELINES, make_problem, reference
from sparsebench.reference import dsd_reference
from tests.cases import CASE_NAMES, routing_case


@pytest.mark.parametrize("case", CASE_NAMES)
def test_layouts_agree_on_cpu(case: str) -> None:
    # fp64 on CPU: proves the grouped <-> padded <-> blocks mapping exactly,
    # without any GPU. Our layout goes through the PyTorch DSD reference.
    p = make_problem(routing_case(case, 37, 8), 8, ffn_dim=32, d_model=24, block_size=16, dtype=torch.float64)
    ref = reference(p)
    ours = dsd_reference(p.blocks, p.w_cat, p.topo)[p.padded_row]  # type: ignore[arg-type]
    torch.testing.assert_close(ours, ref, rtol=1e-12, atol=1e-12)
    for name in ("per-expert cuBLAS loop", "padded bmm"):
        b = next(b for b in BASELINES if b.name == name)
        run, to_grouped = b.setup(p)
        torch.testing.assert_close(to_grouped(run()), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("case", ["random", "zipf", "empty_middle"])
@pytest.mark.parametrize("baseline", BASELINES, ids=[b.name for b in BASELINES])
def test_baseline_correct_on_gpu(baseline: object, case: str) -> None:
    b = baseline
    reason = b.unavailable()  # type: ignore[attr-defined]
    if reason is not None:
        pytest.skip(reason)
    p = make_problem(routing_case(case, 300, 8).cuda(), 8, ffn_dim=256, d_model=256, block_size=128)
    try:
        run, to_grouped = b.setup(p)  # type: ignore[attr-defined]
    except RuntimeError as e:
        if b.name == "torch._grouped_mm":  # type: ignore[attr-defined]
            pytest.skip(f"not supported here: {e}")  # e.g. H100-only in this torch version
        raise
    torch.testing.assert_close(to_grouped(run()).float(), reference(p), rtol=1.6e-2, atol=1.6e-2)
