import pytest
import torch
import torch.nn.functional as F

from sparsebench.reference import group_tokens, sdd_reference
from sparsebench.topology import Topology, build_topology
from tests.cases import CASE_NAMES, routing_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOL = {"rtol": 1.6e-2, "atol": 1.6e-2}


def make_inputs(case: str, num_tokens: int, num_experts: int, d_model: int, ffn_dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = routing_case(case, num_tokens, num_experts, seed).cuda()
    # x ~ N(0, 1), w1 ~ N(0, 1/D): pre-activations are about N(0, 1).
    x = torch.randn(num_tokens, d_model, generator=g).bfloat16().cuda()
    w1 = (torch.randn(num_experts, d_model, ffn_dim, generator=g) / d_model**0.5).bfloat16().cuda()
    return idx, x, w1


def expected_blocks(x: torch.Tensor, w1: torch.Tensor, topo: Topology, gelu: bool) -> torch.Tensor:
    num_experts, d_model, ffn_dim = w1.shape
    w1_cat = w1.float().permute(1, 0, 2).reshape(d_model, num_experts * ffn_dim)
    h = sdd_reference(group_tokens(x.float(), topo), w1_cat, topo)
    return F.gelu(h) if gelu else h


def run_sdd(x, w1, topo, gelu=True):
    from sparsebench.spmm import sdd

    return sdd(x, w1, topo, gelu=gelu)


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize(("num_experts", "d_model", "ffn_dim"), [(8, 128, 256), (8, 320, 512), (64, 64, 256)],
                         ids=["E8D128F256", "E8D320F512", "E64D64F256"])
def test_sdd_matches_reference(case: str, num_experts: int, d_model: int, ffn_dim: int) -> None:
    idx, x, w1 = make_inputs(case, 300, num_experts, d_model, ffn_dim)
    topo = build_topology(idx, num_experts, ffn_dim, 128)
    torch.testing.assert_close(run_sdd(x, w1, topo).float(), expected_blocks(x, w1, topo, True), **TOL)


def test_sdd_without_gelu() -> None:
    idx, x, w1 = make_inputs("zipf", 200, 8, 128, 256)
    topo = build_topology(idx, 8, 256, 128)
    torch.testing.assert_close(run_sdd(x, w1, topo, gelu=False).float(), expected_blocks(x, w1, topo, False), **TOL)


def test_sdd_single_token_is_mostly_padding() -> None:
    # 1 token, 2 copies: each row block is 1 real row + 127 zero-filled ones.
    idx, x, w1 = make_inputs("random", 1, 8, 128, 256)
    topo = build_topology(idx, 8, 256, 128)
    out = run_sdd(x, w1, topo, gelu=False).float()
    torch.testing.assert_close(out, expected_blocks(x, w1, topo, False), **TOL)
    assert torch.all(out[:, 1:, :] == 0), "padding rows must come out exactly zero"


def test_sdd_worst_case_device_topology() -> None:
    # The no-sync path: worst-case buffers straight from route_device.
    from sparsebench.routing import route_device

    idx, x, w1 = make_inputs("zipf", 400, 64, 128, 256)
    dev = route_device(idx, 64, 256, 128)
    out = run_sdd(x, w1, dev)
    topo = dev.to_topology()
    assert out.shape[0] == dev.col_indices.numel()
    torch.testing.assert_close(out[: topo.nnz_blocks].float(), expected_blocks(x, w1, topo, True), **TOL)


BAD = {
    "fp32_x": (lambda x, w1: (x.float(), w1), "bfloat16"),
    "f_not_256": (lambda x, w1: (x, w1[:, :, :128].contiguous()), "multiple of 256"),
    "d_not_64": (lambda x, w1: (x[:, :96].contiguous(), w1[:, :96].contiguous()), "multiple of 64"),
    "d_mismatch": (lambda x, w1: (x[:, :64].contiguous(), w1), r"\[T, D\]"),
}


@pytest.mark.parametrize("name", BAD)
def test_sdd_rejects_bad_input(name: str) -> None:
    from sparsebench import _C

    idx, x, w1 = make_inputs("random", 20, 4, 128, 256)
    topo = build_topology(idx, 4, 256, 128)
    change, match = BAD[name]
    xx, ww = change(x, w1)
    with pytest.raises(RuntimeError, match=match):
        _C.sdd_cuda(xx, ww, topo.row_token, topo.row_block_expert, topo.row_offsets, topo.col_indices.numel(), True)


def test_sdd_rejects_block_size_64() -> None:
    idx, x, w1 = make_inputs("random", 20, 4, 128, 256)
    with pytest.raises(ValueError, match="block size 128"):
        run_sdd(x, w1, build_topology(idx, 4, 256, 64))
