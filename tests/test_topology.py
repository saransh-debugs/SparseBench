import pytest
import torch

from sparsebench.topology import Topology, build_topology, bsr_to_dense, dense_to_bsr
from tests.cases import CASE_NAMES, DEVICES, routing_case


def check_invariants(topo: Topology, expert_idx: torch.Tensor) -> None:
    bs, num_experts = topo.block_size, topo.num_experts
    num_tokens, k = expert_idx.shape
    flat = expert_idx.reshape(-1).cpu().long()
    counts = topo.tokens_per_expert.cpu().long()
    offsets = topo.padded_offsets.cpu().long()
    row_token = topo.row_token.cpu().long()
    copy_row = topo.copy_row.cpu().long()

    assert torch.equal(counts, torch.bincount(flat, minlength=num_experts))

    # Each expert's group: starts at 0, padded to a multiple of bs, wastes < bs rows.
    sizes = offsets[1:] - offsets[:-1]
    assert offsets[0] == 0
    assert torch.all(sizes % bs == 0)
    assert torch.all(sizes >= counts) and torch.all(sizes - counts < bs)
    assert torch.all(sizes[counts == 0] == 0)

    # Every copy lands on its own row, inside its expert's range, pointing back to its token.
    assert copy_row.unique().numel() == num_tokens * k
    assert torch.all(copy_row >= offsets[flat]) and torch.all(copy_row < offsets[flat] + counts[flat])
    assert torch.equal(row_token[copy_row], torch.arange(num_tokens * k) // k)
    assert int((row_token < 0).sum()) == topo.num_rows - num_tokens * k

    # Stable order: inside one expert, rows follow copy order.
    for e in range(num_experts):
        rows = copy_row[flat == e]
        assert torch.all(rows[1:] > rows[:-1])

    # BSR: row block r covers rows [r*bs, (r+1)*bs) of its expert, and has
    # exactly that expert's F column blocks.
    rbe = topo.row_block_expert.cpu().long()
    starts = torch.arange(topo.num_row_blocks) * bs
    assert torch.all(starts >= offsets[rbe]) and torch.all(starts + bs <= offsets[rbe + 1])
    assert torch.all(topo.row_offsets.cpu().diff() == topo.ffn_blocks)
    cols = topo.col_indices.cpu().long().view(topo.num_row_blocks, topo.ffn_blocks)
    expected = rbe[:, None] * topo.ffn_blocks + torch.arange(topo.ffn_blocks)
    assert torch.equal(cols, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize(
    ("num_tokens", "block_size"),
    [(1, 4), (7, 4), (16, 4), (300, 16), (1000, 128)],
)
def test_invariants(device: str, case: str, num_tokens: int, block_size: int) -> None:
    num_experts = 8
    expert_idx = routing_case(case, num_tokens, num_experts).to(device)
    topo = build_topology(expert_idx, num_experts, ffn_dim=2 * block_size, block_size=block_size)
    check_invariants(topo, expert_idx)


def test_exact_multiple_has_no_padding() -> None:
    # 8 tokens * 2 copies over 2 experts = 8 per expert = 2 blocks of 4. No padding rows.
    expert_idx = torch.tensor([[0, 1]]).repeat(8, 1)
    topo = build_topology(expert_idx, num_experts=4, ffn_dim=8, block_size=4)
    assert topo.num_rows == 16
    assert int((topo.row_token < 0).sum()) == 0


def test_int32_input_matches_int64() -> None:
    expert_idx = routing_case("random", 50, 8)
    a = build_topology(expert_idx, 8, 32, 16)
    b = build_topology(expert_idx.int(), 8, 32, 16)
    assert torch.equal(a.copy_row, b.copy_row) and torch.equal(a.row_token, b.row_token)


@pytest.mark.parametrize(
    ("expert_idx", "kwargs", "error"),
    [
        (torch.zeros(0, 2, dtype=torch.long), {}, ValueError),
        (torch.zeros(4, dtype=torch.long), {}, ValueError),
        (torch.zeros(4, 2), {}, TypeError),
        (torch.tensor([[0, 8]]), {}, ValueError),
        (torch.tensor([[-1, 0]]), {}, ValueError),
        (torch.tensor([[0, 1]]), {"ffn_dim": 10}, ValueError),
        (torch.tensor([[0, 1]]), {"block_size": 0}, ValueError),
    ],
)
def test_rejects_bad_input(expert_idx: torch.Tensor, kwargs: dict[str, int], error: type) -> None:
    args = {"num_experts": 8, "ffn_dim": 8, "block_size": 4} | kwargs
    with pytest.raises(error):
        build_topology(expert_idx, **args)


@pytest.mark.parametrize("device", DEVICES)
def test_bsr_round_trip(device: str) -> None:
    topo = build_topology(routing_case("skewed", 37, 4).to(device), 4, ffn_dim=8, block_size=4)
    blocks = torch.randn(topo.nnz_blocks, 4, 4, dtype=torch.float64, device=device)
    dense = bsr_to_dense(blocks, topo)
    assert torch.equal(dense_to_bsr(dense, topo), blocks)
    # Everything outside the layout is zero: total mass matches the blocks alone.
    assert torch.equal(dense.abs().sum(), blocks.abs().sum())
