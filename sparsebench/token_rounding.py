"""Token rounding (SonicMoE, Guo et al., ICLR 2026, Algorithm 4 / Section 5.2).

Block-sparse and grouped-GEMM MoE kernels work in tiles of 128 rows, so each
expert's token count gets rounded up to a multiple of 128 and the extra rows
are wasted math. Token rounding moves that rounding into the router instead:

  1. Normal top-k routing; f_e = how many tokens picked expert e.
  2. Round f_e to the NEAREST multiple of the tile (ties round down, the
     paper's rule: pad only if ceil - f < f - floor).
  3. Rounding down drops expert e's lowest-scoring top-k tokens. Rounding up
     adds the highest-scoring tokens that did NOT pick e. Scores are shifted
     (S - 1, top-k entries restored) so top-k picks always rank first: only
     the last tile of each expert changes.
  4. Each token's weights are renormalized over the experts it kept.

Each expert moves by at most half a tile, but a token can end up with fewer
or more than k experts, even 0. The paper trains with token rounding and
evaluates with plain top-k, and reports it robust when experts average at
least 2 tiles of tokens.

Plain PyTorch, vectorized, CPU or GPU. One host sync (nonzero) to get the
number of kept assignments, like any variable-size router.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Assignments:
    """A routing as a flat list: copy c sends token[c] to expert[c] with weight[c]."""

    token: torch.Tensor  # [N] int64
    expert: torch.Tensor  # [N] int64
    weight: torch.Tensor  # [N] float32


def top_k_assignments(scores: torch.Tensor, k: int) -> Assignments:
    """Plain top-k, as a flat list. Weights: softmax over the k picked scores,
    i.e. the picked scores renormalized to sum to 1 (Mixtral style)."""
    top_s, top_i = scores.topk(k, dim=1)
    num_tokens = scores.shape[0]
    token = torch.arange(num_tokens, device=scores.device).repeat_interleave(k)
    weight = (top_s / top_s.sum(dim=1, keepdim=True)).reshape(-1)
    return Assignments(token, top_i.reshape(-1), weight.float())


def round_counts(counts: torch.Tensor, tile: int, max_count: int) -> torch.Tensor:
    """Nearest multiple of `tile`; ties round down. Never more than max_count
    (the largest multiple of the tile that fits in the token count)."""
    low = counts // tile * tile
    high = low + tile * (counts % tile > 0)
    rounded = torch.where(high - counts < counts - low, high, low)
    return rounded.clamp(max=max_count // tile * tile)


def token_rounding(scores: torch.Tensor, k: int, tile: int = 128) -> Assignments:
    """scores [T, E] router probabilities -> routing where every expert's
    token count is a multiple of `tile`."""
    num_tokens, num_experts = scores.shape
    top_s, top_i = scores.topk(k, dim=1)
    counts = torch.zeros(num_experts, dtype=torch.long, device=scores.device)
    counts.scatter_add_(0, top_i.reshape(-1), torch.ones_like(top_i.reshape(-1)))
    target = round_counts(counts, tile, num_tokens)

    # Top-k picks keep their score; everything else drops by 1, so within an
    # expert's column every pick outranks every non-pick.
    shifted = scores - 1
    shifted.scatter_(1, top_i, top_s)
    # rank[t, e] = position of token t in expert e's column, best first.
    # Stable sort: equal scores keep token order, so results are deterministic.
    order = torch.sort(shifted, dim=0, descending=True, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(num_tokens, device=scores.device)[:, None].expand_as(order))
    keep = rank < target[None, :]

    token, expert = keep.nonzero(as_tuple=True)  # the one host sync
    raw = scores[token, expert]
    total = torch.zeros(num_tokens, dtype=raw.dtype, device=scores.device).index_add_(0, token, raw)
    return Assignments(token, expert, (raw / total[token]).float())


def routing_change(scores: torch.Tensor, k: int, rounded: Assignments, tile: int = 128) -> dict[str, float]:
    """How far `rounded` is from plain top-k, and how much padding top-k needs.
    Syncs (returns Python numbers): for reporting, not for the hot path."""
    num_tokens, num_experts = scores.shape
    top_s, top_i = scores.topk(k, dim=1)
    picked = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top_i, True)
    kept = torch.zeros_like(picked)
    kept[rounded.token, rounded.expert] = True

    counts = picked.sum(dim=0)
    padded = ((counts + tile - 1) // tile * tile).sum()
    experts_per_token = kept.sum(dim=1)
    return {
        "topk_padding_pct": 100 * float((padded - counts.sum()) / padded),
        "topk_picks_dropped_pct": 100 * float((picked & ~kept).sum() / picked.sum()),
        "assignments_added_pct": 100 * float((kept & ~picked).sum() / picked.sum()),
        "tokens_with_0_experts_pct": 100 * float((experts_per_token == 0).float().mean()),
        "tokens_with_more_than_k_pct": 100 * float((experts_per_token > k).float().mean()),
        "topk_score_mass_kept_pct": 100 * float(scores[picked & kept].sum() / scores[picked].sum()),
    }
