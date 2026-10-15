import dataclasses

import pytest
import torch

from sparsebench.topology import Topology, build_topology
from tests.cases import CASE_NAMES, routing_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def route_cuda(expert_idx: torch.Tensor, num_experts: int, ffn_dim: int, block_size: int) -> Topology:
    from sparsebench.routing import build_topology_cuda

    return build_topology_cuda(expert_idx, num_experts, ffn_dim, block_size)


def assert_same(actual: Topology, expected: Topology) -> None:
    for field in dataclasses.fields(Topology):
        a, e = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(e, torch.Tensor):
            assert a.dtype == e.dtype and a.device == e.device, field.name
            assert torch.equal(a, e), f"{field.name} differs"
        else:
            assert a == e, field.name


def random_topk(num_tokens: int, num_experts: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(num_tokens, num_experts, generator=g).topk(k, dim=-1).indices


def check_against_reference(expert_idx: torch.Tensor, num_experts: int, block_size: int) -> None:
    ffn_dim = 2 * block_size
    expert_idx = expert_idx.cuda()
    expected = build_topology(expert_idx, num_experts, ffn_dim, block_size)
    assert_same(route_cuda(expert_idx, num_experts, ffn_dim, block_size), expected)


# Token counts hit: 1 copy (k=1), not a multiple of 32, crosses one 256-copy
# chunk, many chunks.
@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("num_experts", [2, 8, 64, 256])
@pytest.mark.parametrize("block_size", [4, 16, 128])
@pytest.mark.parametrize("num_tokens", [1, 13, 129, 1000])
def test_matches_reference(case: str, num_experts: int, block_size: int, num_tokens: int) -> None:
    check_against_reference(routing_case(case, num_tokens, num_experts), num_experts, block_size)


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 77, 3000])
def test_matches_reference_any_k(k: int, num_tokens: int) -> None:
    check_against_reference(random_topk(num_tokens, 16, k), 16, 16)


@pytest.mark.parametrize("case", ["random", "skewed"])
@pytest.mark.parametrize("num_experts", [8, 1024])
def test_matches_reference_big(case: str, num_experts: int) -> None:
    # 200k copies = 782 thread blocks. Skipped under compute-sanitizer (slow).
    check_against_reference(routing_case(case, 100_000, num_experts), num_experts, 128)


def test_int32_and_int64_input_agree() -> None:
    expert_idx = routing_case("random", 500, 8).cuda()
    assert_same(route_cuda(expert_idx.int(), 8, 32, 16), route_cuda(expert_idx, 8, 32, 16))


def test_deterministic() -> None:
    expert_idx = routing_case("skewed", 5000, 64).cuda()
    assert_same(route_cuda(expert_idx, 64, 256, 128), route_cuda(expert_idx, 64, 256, 128))


def test_worst_case_buffers_are_marked_past_the_end() -> None:
    from sparsebench.routing import route_device

    dev = route_device(routing_case("random", 100, 8).cuda(), 8, 32, 16)
    num_rows = int(dev.padded_offsets[-1])
    num_row_blocks = num_rows // 16
    assert torch.all(dev.row_token[num_rows:] == -1)
    assert torch.all(dev.row_block_expert[num_row_blocks:] == -1)
    assert torch.all(dev.col_indices[num_row_blocks * 2 :] == -1)
    # Empty rows past the end: row_offsets stays flat at nnz.
    assert torch.all(dev.row_offsets[num_row_blocks:] == num_row_blocks * 2)


def test_cuda_graph_capture_proves_no_host_sync() -> None:
    # Graph capture fails if the captured code waits on the GPU. Replaying with
    # new routing also proves the graph reads its input, not a baked-in result.
    from sparsebench.routing import route_device

    num_tokens, num_experts, ffn_dim, block_size = 4000, 32, 256, 128
    static_idx = routing_case("random", num_tokens, num_experts, seed=1).cuda()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside capture, as torch requires
        route_device(static_idx, num_experts, ffn_dim, block_size)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route_device(static_idx, num_experts, ffn_dim, block_size)

    for case in ["skewed", "empty_middle"]:
        new_idx = routing_case(case, num_tokens, num_experts, seed=2).cuda()
        static_idx.copy_(new_idx)
        graph.replay()
        assert_same(captured.to_topology(), build_topology(new_idx, num_experts, ffn_dim, block_size))


def test_out_of_range_id_raises_without_writing_out_of_bounds() -> None:
    from sparsebench.routing import route_device

    expert_idx = torch.tensor([[0, 1], [2, 8], [-1, 3]], device="cuda")
    dev = route_device(expert_idx, 8, 32, 16)
    with pytest.raises(ValueError, match="outside"):
        dev.to_topology()
    # The two bad copies were skipped, the four good ones were placed.
    assert int(dev.tokens_per_expert.sum()) == 4
    assert dev.copy_row.tolist() == [0, 16, 32, -1, -1, 48]


BAD_INPUTS = {
    "cpu_tensor": (lambda: torch.tensor([[0, 1]]), 8, 32, "CUDA"),
    "not_2d": (lambda: torch.tensor([0, 1], device="cuda"), 8, 32, r"\[T, k\]"),
    "too_many_experts": (lambda: torch.tensor([[0, 1]], device="cuda"), 2000, 32, "num_experts"),
    "ffn_not_multiple": (lambda: torch.tensor([[0, 1]], device="cuda"), 8, 30, "multiple"),
    "float_ids": (lambda: torch.tensor([[0.0, 1.0]], device="cuda"), 8, 32, "int32"),
}


@pytest.mark.parametrize("name", BAD_INPUTS)
def test_rejects_bad_input(name: str) -> None:
    make, num_experts, ffn_dim, match = BAD_INPUTS[name]
    with pytest.raises(RuntimeError, match=match):
        route_cuda(make(), num_experts, ffn_dim, 16)
