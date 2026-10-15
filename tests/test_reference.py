import pytest
import torch
import torch.nn.functional as F

from sparsebench.reference import moe_blocksparse_reference, moe_reference, route
from tests.cases import CASE_NAMES, DEVICES, routing_case


def make_weights(
    num_experts: int, d_model: int, ffn_dim: int, device: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(num_experts, d_model, ffn_dim, generator=g, dtype=torch.float64) / d_model**0.5
    w2 = torch.randn(num_experts, ffn_dim, d_model, generator=g, dtype=torch.float64) / ffn_dim**0.5
    return w1.to(device), w2.to(device)


def per_token_oracle(
    x: torch.Tensor, expert_idx: torch.Tensor, expert_weight: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """One token at a time. Slow, but nothing can go wrong with grouping."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(expert_idx.shape[1]):
            e = int(expert_idx[t, j])
            out[t] += expert_weight[t, j] * (F.gelu(x[t] @ w1[e]) @ w2[e])
    return out


def test_route_picks_top_k_and_weights_sum_to_one() -> None:
    logits = torch.tensor([[0.1, 3.0, 2.0, -1.0], [5.0, 4.0, 0.0, 0.0]])
    expert_idx, weight = route(logits, k=2)
    assert expert_idx.tolist() == [[1, 2], [0, 1]]
    assert torch.allclose(weight.sum(-1), torch.ones(2))
    assert torch.all(weight[:, 0] >= weight[:, 1])


@pytest.mark.parametrize("case", CASE_NAMES)
def test_loop_reference_matches_per_token_oracle(case: str) -> None:
    num_experts, d_model, ffn_dim = 6, 8, 12
    expert_idx = routing_case(case, 13, num_experts)
    x = torch.randn(13, d_model, dtype=torch.float64)
    weight = torch.rand(13, 2, dtype=torch.float64)
    w1, w2 = make_weights(num_experts, d_model, ffn_dim, "cpu")
    torch.testing.assert_close(
        moe_reference(x, expert_idx, weight, w1, w2), per_token_oracle(x, expert_idx, weight, w1, w2)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize(("num_tokens", "block_size"), [(1, 4), (29, 4), (200, 16), (300, 64)])
def test_blocksparse_matches_loop(device: str, case: str, num_tokens: int, block_size: int) -> None:
    num_experts, d_model = 8, 24
    ffn_dim = 2 * block_size
    expert_idx = routing_case(case, num_tokens, num_experts).to(device)
    x = torch.randn(num_tokens, d_model, dtype=torch.float64, device=device)
    weight = torch.rand(num_tokens, 2, dtype=torch.float64, device=device)
    w1, w2 = make_weights(num_experts, d_model, ffn_dim, device)

    expected = moe_reference(x, expert_idx, weight, w1, w2)
    actual = moe_blocksparse_reference(x, expert_idx, weight, w1, w2, block_size)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_routed_end_to_end() -> None:
    # Router output feeds straight into both paths.
    num_experts, d_model, ffn_dim = 8, 16, 32
    x = torch.randn(64, d_model, dtype=torch.float64)
    expert_idx, weight = route(x @ torch.randn(d_model, num_experts, dtype=torch.float64))
    w1, w2 = make_weights(num_experts, d_model, ffn_dim, "cpu")
    torch.testing.assert_close(
        moe_blocksparse_reference(x, expert_idx, weight, w1, w2, block_size=16),
        moe_reference(x, expert_idx, weight, w1, w2),
    )


def test_rejects_mismatched_shapes() -> None:
    w1, w2 = make_weights(4, 8, 16, "cpu")
    x = torch.randn(5, 8, dtype=torch.float64)
    idx = torch.tensor([[0, 1]]).repeat(5, 1)
    weight = torch.ones(5, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="w2"):
        moe_reference(x, idx, weight, w1, w2.transpose(1, 2))
    with pytest.raises(ValueError, match="expert_weight"):
        moe_reference(x, idx, weight[:, :1], w1, w2)
    with pytest.raises(ValueError, match="x must be"):
        moe_reference(x[None], idx, weight, w1, w2)
