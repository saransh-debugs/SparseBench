import math

import pytest
import torch

from sparsebench.reference import dsd_reference
from sparsebench.topology import Topology, build_topology
from tests.cases import CASE_NAMES, routing_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# 21 / 31 = v2 / v3 with a 2-stage pipeline. Only the benchmark uses them,
# but a shorter ring is the likeliest to have a stage-reuse bug, so test them too.
VERSIONS = [pytest.param(v, id=f"v{v}") for v in (1, 2, 21, 3, 31)]
TOL = {"rtol": 1.6e-2, "atol": 1.6e-2}  # bf16 output rounding; layout bugs are far bigger


def bsr_matmul_oracle(blocks: torch.Tensor, w: torch.Tensor, topo: Topology) -> torch.Tensor:
    """BSR @ dense straight from the definition, in fp32. Works for any BSR
    layout, and uses far less memory than dsd_reference's dense H."""
    bs = topo.block_size
    nnz = topo.nnz_blocks
    w_blocks = w.float().view(-1, bs, w.shape[1])[topo.col_indices[:nnz].long()]  # [nnz, bs, D]
    products = torch.bmm(blocks[:nnz].float(), w_blocks)
    counts = (topo.row_offsets[1:] - topo.row_offsets[:-1]).long()
    row_of_block = torch.repeat_interleave(torch.arange(topo.num_row_blocks, device=w.device), counts)
    out = torch.zeros(topo.num_row_blocks, bs, w.shape[1], device=w.device)
    out.index_add_(0, row_of_block, products)
    return out.view(-1, w.shape[1])


def make_inputs(
    case: str, num_tokens: int, num_experts: int, ffn_dim: int, d_model: int, bs: int, seed: int = 0
) -> tuple[Topology, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    topo = build_topology(routing_case(case, num_tokens, num_experts, seed).cuda(), num_experts, ffn_dim, bs)
    # H ~ N(0, 1/F), W ~ N(0, 1): each output is a sum of F terms, so about N(0, 1).
    blocks = (torch.randn(topo.nnz_blocks, bs, bs, generator=g) / math.sqrt(ffn_dim)).bfloat16().cuda()
    w = torch.randn(num_experts * ffn_dim, d_model, generator=g).bfloat16().cuda()
    return topo, blocks, w


def run(blocks: torch.Tensor, w: torch.Tensor, topo: object, version: int) -> torch.Tensor:
    from sparsebench.spmm import dsd

    return dsd(blocks, w, topo, version=version)  # type: ignore[arg-type]


def test_oracle_matches_dense_reference() -> None:
    # Ties the cheap oracle to dsd_reference (dense H) on a small case.
    topo, blocks, w = make_inputs("zipf", 50, 4, 128, 128, 64)
    torch.testing.assert_close(
        bsr_matmul_oracle(blocks, w, topo), dsd_reference(blocks.float(), w.float(), topo), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("bs", [64, 128])
@pytest.mark.parametrize(("num_experts", "f_blocks"), [(8, 1), (8, 4), (64, 1), (64, 2)], ids=["E8F1", "E8F4", "E64F1", "E64F2"])
@pytest.mark.parametrize("d_model", [128, 512], ids=["D128", "D512"])
def test_matches_oracle(version: int, case: str, bs: int, num_experts: int, f_blocks: int, d_model: int) -> None:
    # bs=64 with F1 gives rows of only 2 K tiles, fewer than v2's 3 stages.
    # Dead and empty rows (0 tiles) come from the worst-case test below.
    topo, blocks, w = make_inputs(case, 300, num_experts, f_blocks * bs, d_model, bs)
    torch.testing.assert_close(run(blocks, w, topo, version).float(), bsr_matmul_oracle(blocks, w, topo), **TOL)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("bs", [64, 128])
def test_single_token(version: int, bs: int) -> None:
    topo, blocks, w = make_inputs("random", 1, 8, 2 * bs, 128, bs)
    torch.testing.assert_close(run(blocks, w, topo, version).float(), bsr_matmul_oracle(blocks, w, topo), **TOL)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("bs", [64, 128])
def test_asymmetric_inputs(version: int, bs: int) -> None:
    # Smooth, structured values with no symmetry between rows and columns
    # (or between H and W), so a transposed or swapped fragment can't hide.
    topo, _, _ = make_inputs("random", 200, 8, 2 * bs, 256, bs)
    k_total, d_model = 8 * 2 * bs, 256
    i = torch.arange(k_total, dtype=torch.float32)[:, None]
    j = torch.arange(d_model, dtype=torch.float32)[None, :]
    w = torch.sin(0.37 * i + 1.31 * j + 0.05 * i * j / d_model).bfloat16().cuda()
    p = torch.arange(topo.nnz_blocks, dtype=torch.float32)[:, None, None]
    r = torch.arange(bs, dtype=torch.float32)[None, :, None]
    c = torch.arange(bs, dtype=torch.float32)[None, None, :]
    blocks = (torch.cos(0.7 * r - 0.23 * c + 0.11 * p + 0.01 * r * c) / math.sqrt(2 * bs)).bfloat16().cuda()
    torch.testing.assert_close(run(blocks, w, topo, version).float(), bsr_matmul_oracle(blocks, w, topo), **TOL)


@pytest.mark.parametrize("version", VERSIONS)
def test_worst_case_device_topology(version: int) -> None:
    # The no-sync path: route_device's worst-case buffers go straight in.
    from sparsebench.routing import route_device

    bs, num_experts, ffn_dim, d_model = 128, 64, 256, 256
    dev = route_device(routing_case("zipf", 400, num_experts).cuda(), num_experts, ffn_dim, bs)
    g = torch.Generator().manual_seed(3)
    blocks = (torch.randn(dev.col_indices.numel(), bs, bs, generator=g) / 16).bfloat16().cuda()
    w = torch.randn(num_experts * ffn_dim, d_model, generator=g).bfloat16().cuda()

    out = run(blocks, w, dev, version)
    topo = dev.to_topology()
    assert out.shape == ((dev.row_offsets.numel() - 1) * bs, d_model)
    torch.testing.assert_close(out[: topo.num_rows].float(), bsr_matmul_oracle(blocks, w, topo), **TOL)
    assert torch.all(out[topo.num_rows :] == 0)


BAD_INPUTS = {
    "fp32_weights": (lambda b, w: (b, w.float()), "bfloat16"),
    "fp16_blocks": (lambda b, w: (b.half(), w), "bfloat16"),
    "d_not_multiple": (lambda b, w: (b, w[:, :100].contiguous()), "multiple of 128"),
    "non_contiguous": (lambda b, w: (b, w.t().contiguous().t()), "contiguous"),
    "cpu_weights": (lambda b, w: (b, w.cpu()), "CUDA"),
}


@pytest.mark.parametrize("name", BAD_INPUTS)
def test_rejects_bad_input(name: str) -> None:
    topo, blocks, w = make_inputs("random", 20, 4, 128, 256, 64)
    change, match = BAD_INPUTS[name]
    b, w2 = change(blocks, w)
    with pytest.raises(RuntimeError, match=match):
        run(b, w2, topo, 1)


def test_rejects_unsupported_block_size_and_version() -> None:
    from sparsebench import _C

    topo, blocks, w = make_inputs("random", 20, 4, 64, 256, 64)
    with pytest.raises(RuntimeError, match="unknown dsd version"):
        run(blocks, w, topo, 99)
    topo32 = build_topology(routing_case("random", 20, 4).cuda(), 4, 64, 32)
    blocks32 = torch.zeros(topo32.nnz_blocks, 32, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="64 or 128"):
        _C.dsd_cuda(blocks32, w, topo32.row_offsets, topo32.col_indices, 1)


def test_rejects_block_shape_mismatch() -> None:
    topo, blocks, w = make_inputs("random", 20, 4, 128, 256, 64)
    with pytest.raises(ValueError, match="blocks must be"):
        run(blocks[:, :32], w, topo, 1)
