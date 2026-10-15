"""Routing patterns shared by the tests. Includes the ones that break kernels."""

import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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
    raise ValueError(name)


CASE_NAMES = ["random", "skewed", "two_experts_only", "empty_middle"]
