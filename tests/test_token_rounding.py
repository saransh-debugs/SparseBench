import pytest
import torch
import torch.nn.functional as F

from sparsebench.token_rounding import round_counts, routing_change, token_rounding, top_k_assignments
from sparsebench.workloads import router_scores


def counts_of(expert: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(expert, minlength=num_experts)


def test_round_counts_nearest_ties_down() -> None:
    counts = torch.tensor([0, 1, 63, 64, 65, 128, 191, 192, 200])
    expected = torch.tensor([0, 0, 0, 0, 128, 128, 128, 128, 256])
    assert torch.equal(round_counts(counts, 128, 10_000), expected)


def test_hand_worked_case() -> None:
    # 8 tokens, 2 experts, k = 1, tile 4. Tokens 0-4 pick expert 0 (f = 5),
    # tokens 5-7 pick expert 1 (f = 3).
    # Expert 0: 5 -> nearest multiple of 4 is 4: drop its lowest-scoring pick.
    # Expert 1: 3 -> 4: add the best non-pick, ranked by expert-1 score.
    s0 = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.2, 0.3])
    scores = torch.stack([s0, 1 - s0], dim=1)
    a = token_rounding(scores, k=1, tile=4)
    kept = set(zip(a.token.tolist(), a.expert.tolist()))
    # Expert 0 drops token 4 (score 0.55, its lowest pick).
    assert {t for t, e in kept if e == 0} == {0, 1, 2, 3}
    # Expert 1 keeps 5, 6, 7 and adds token 4 (expert-1 score 0.45, the best non-pick).
    assert {t for t, e in kept if e == 1} == {4, 5, 6, 7}
    # Token 4 moved experts entirely; everyone still has exactly one, weight 1.
    assert torch.allclose(a.weight, torch.ones_like(a.weight))


@pytest.mark.parametrize("name", ["uniform", "zipf"])
@pytest.mark.parametrize(("num_tokens", "num_experts", "k"), [(4096, 64, 2), (2000, 8, 2), (3000, 16, 4)])
def test_properties(name: str, num_tokens: int, num_experts: int, k: int) -> None:
    tile = 128
    scores = router_scores(name, num_tokens, num_experts)
    rounded = token_rounding(scores, k, tile)
    plain = top_k_assignments(scores, k)
    before = counts_of(plain.expert, num_experts)
    after = counts_of(rounded.expert, num_experts)

    # Every expert gets a multiple of the tile: no padding needed.
    assert torch.all(after % tile == 0)
    # Nearest rounding: each expert moves by at most half a tile.
    assert torch.all((after - before).abs() <= tile // 2)

    picked = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    picked[plain.token, plain.expert] = True
    kept = torch.zeros_like(picked)
    kept[rounded.token, rounded.expert] = True
    for e in range(num_experts):
        col = scores[:, e]
        if after[e] <= before[e]:
            # Rounded down: only top-k picks kept, and they're the best-scoring ones.
            assert not (kept[:, e] & ~picked[:, e]).any()
            # (An expert rounded down to 0 keeps nothing: Zipf tail experts with
            # under half a tile of tokens.)
            if kept[:, e].any() and (picked[:, e] & ~kept[:, e]).any():
                assert col[kept[:, e]].min() >= col[picked[:, e] & ~kept[:, e]].max()
        else:
            # Rounded up: all picks kept, plus the best-scoring non-picks.
            assert torch.equal(kept[:, e] & picked[:, e], picked[:, e])
            added = kept[:, e] & ~picked[:, e]
            rest = ~kept[:, e]
            if rest.any():
                assert col[added].min() >= col[rest].max()

    # Weights sum to 1 for every token that kept at least one expert.
    sums = torch.zeros(num_tokens).index_add_(0, rounded.token, rounded.weight)
    has = torch.zeros(num_tokens, dtype=torch.bool)
    has[rounded.token] = True
    torch.testing.assert_close(sums[has], torch.ones(int(has.sum())))

    # Deterministic.
    again = token_rounding(scores, k, tile)
    assert torch.equal(again.token, rounded.token) and torch.equal(again.expert, rounded.expert)

    stats = routing_change(scores, k, rounded, tile)
    assert 0 <= stats["topk_picks_dropped_pct"] <= 100


# ---- the layer on a flat assignment list (GPU) ---------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference_assigned(x, token, expert, weight, w1, w2) -> torch.Tensor:
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device)
    for e in range(w1.shape[0]):
        m = expert == e
        if m.any():
            y = F.gelu(x[token[m]].float() @ w1[e].float()) @ w2[e].float()
            out.index_add_(0, token[m], y * weight[m, None].float())
    return out


def make_layer(num_tokens: int, num_experts: int, d_model: int = 256, ffn_dim: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, d_model, generator=g).bfloat16().cuda()
    w1 = (torch.randn(num_experts, d_model, ffn_dim, generator=g) / d_model**0.5).bfloat16().cuda()
    w2 = (torch.randn(num_experts, ffn_dim, d_model, generator=g) / ffn_dim**0.5).bfloat16().cuda()
    return x, w1, w2


@gpu
@pytest.mark.parametrize("name", ["uniform", "zipf"])
def test_moe_assigned_with_token_rounding(name: str) -> None:
    from sparsebench.moe import moe_forward_assigned

    num_tokens, num_experts = 2048, 16
    x, w1, w2 = make_layer(num_tokens, num_experts)
    a = token_rounding(router_scores(name, num_tokens, num_experts).cuda(), 2)
    out = moe_forward_assigned(x, a.token, a.expert, a.weight, w1, w2)
    expected = reference_assigned(x, a.token, a.expert, a.weight, w1, w2)
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@gpu
def test_moe_assigned_handles_0_and_3_experts_per_token() -> None:
    from sparsebench.moe import moe_forward_assigned

    x, w1, w2 = make_layer(6, 4)
    # Token 0: none. Token 1: one. Token 2: three. Tokens 3-5: two each.
    token = torch.tensor([1, 2, 2, 2, 3, 3, 4, 4, 5, 5], device="cuda")
    expert = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 3], device="cuda")
    weight = torch.rand(10, generator=torch.Generator().manual_seed(1)).cuda()
    out = moe_forward_assigned(x, token, expert, weight, w1, w2)
    torch.testing.assert_close(out.float(), reference_assigned(x, token, expert, weight, w1, w2), rtol=3e-2, atol=3e-2)
    assert torch.all(out[0] == 0)


@gpu
def test_moe_assigned_equals_moe_forward_for_plain_top_k() -> None:
    from sparsebench.moe import moe_forward, moe_forward_assigned

    num_tokens, num_experts = 1000, 8
    x, w1, w2 = make_layer(num_tokens, num_experts)
    scores = router_scores("zipf", num_tokens, num_experts).cuda()
    top_s, top_i = scores.topk(2, dim=1)
    weight = (top_s / top_s.sum(1, keepdim=True)).bfloat16()
    a = top_k_assignments(scores, 2)
    expected = moe_forward(x, top_i, weight, w1, w2)
    got = moe_forward_assigned(x, a.token, a.expert, weight.reshape(-1), w1, w2)
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
