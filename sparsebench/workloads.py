"""Routing patterns (expert_idx tensors) for tests and benchmarks.

Includes the ones that break kernels: empty experts, one overloaded expert,
and Zipf-skewed routing like real MoE routers produce.
"""

import torch

ZIPF_EXPONENT = 1.2


def routing_case(name: str, num_tokens: int, num_experts: int, seed: int = 0) -> torch.Tensor:
    """Returns expert_idx [T, 2], int64, two different experts per token."""
    g = torch.Generator().manual_seed(seed)
    if name == "random":
        return torch.rand(num_tokens, num_experts, generator=g).topk(2, dim=-1).indices
    if name == "skewed":
        # Every token's first choice is expert 0. The rest spread out.
        second = torch.randint(1, num_experts, (num_tokens,), generator=g)
        return torch.stack([torch.zeros_like(second), second], dim=1)
    if name == "two_experts_only":
        # All other experts get zero tokens.
        return torch.tensor([[0, 1]]).repeat(num_tokens, 1)
    if name == "empty_middle":
        # Experts 1..E-2 get nothing. Catches offset bugs around empty groups.
        idx = torch.tensor([[0, num_experts - 1]]).repeat(num_tokens, 1)
        return idx[:, torch.randperm(2, generator=g)]
    if name == "zipf":
        # Expert popularity falls off as 1 / rank^1.2, like real MoE routers
        # that collapse onto a few experts. 2 distinct experts per token.
        weights = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64) ** ZIPF_EXPONENT
        return torch.multinomial(weights.expand(num_tokens, -1), 2, replacement=False, generator=g)
    raise ValueError(name)


CASE_NAMES = ["random", "skewed", "two_experts_only", "empty_middle", "zipf"]
