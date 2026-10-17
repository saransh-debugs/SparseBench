"""A full MoE layer forward pass on our kernels, with no host sync anywhere.

    routing (routing.cu) -> SDD + GELU (sdd.cu) -> DSD v5 (dsd_v5.cu) -> combine

Every buffer is sized for the worst case on the host, so the CPU never waits
for the GPU to learn how many rows routing produced. Same math as
reference.moe_reference.
"""

import dataclasses
from collections.abc import Callable

import torch

from sparsebench.routing import route_device
from sparsebench.spmm import dsd, sdd

BLOCK_SIZE = 128


def _no_mark(stage: str) -> None:
    pass


def moe_forward(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    expert_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    mark: Callable[[str], None] = _no_mark,
) -> torch.Tensor:
    """x [T, D], expert_idx / expert_weight [T, k], w1 [E, D, F], w2 [E, F, D] -> [T, D].
    bf16 weights and activations; fp32 accumulation inside the kernels.

    mark(stage) is called right after each stage is launched; the per-stage
    benchmark (bench/bench_breakdown.py) records a CUDA event there."""
    num_experts, d_model, ffn_dim = w1.shape
    if w2.shape != (num_experts, ffn_dim, d_model):
        raise ValueError(f"w2 must be {(num_experts, ffn_dim, d_model)}, got {tuple(w2.shape)}")
    if x.shape[1] != d_model or expert_idx.shape != expert_weight.shape or expert_idx.shape[0] != x.shape[0]:
        raise ValueError("x must be [T, D] and expert_idx / expert_weight [T, k]")

    topo = route_device(expert_idx, num_experts, ffn_dim, BLOCK_SIZE)
    mark("routing")
    h = sdd(x, w1, topo, gelu=True)
    mark("SDD + GELU (fused)")
    y = dsd(h, w2.view(num_experts * ffn_dim, d_model), topo, version=5)
    mark("DSD")

    # Each token's k rows, weighted. Indexing with GPU tensors: no sync.
    num_tokens, k = expert_idx.shape
    rows = y[topo.copy_row.long()].view(num_tokens, k, d_model)
    out = (rows.float() * expert_weight.float()[..., None]).sum(dim=1).to(x.dtype)
    mark("combine")
    return out


def moe_forward_assigned(
    x: torch.Tensor,
    token_of_copy: torch.Tensor,
    expert_of_copy: torch.Tensor,
    weight_of_copy: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Same layer for any routing given as a flat list: copy c sends token
    token_of_copy[c] to expert expert_of_copy[c] with weight weight_of_copy[c].
    A token may have any number of copies, including none (output row = 0),
    e.g. after token rounding (sparsebench/token_rounding.py).

    No kernel changes: the copies go through route_device as k = 1, and
    row_token is remapped from copy index to token. Still no host sync.
    """
    num_experts, d_model, ffn_dim = w1.shape
    num_tokens = x.shape[0]
    if expert_of_copy.numel() == 0:  # host-known size: nothing routed anywhere
        return torch.zeros_like(x)

    topo = route_device(expert_of_copy.view(-1, 1), num_experts, ffn_dim, BLOCK_SIZE)
    row_copy = topo.row_token  # with k = 1 these are copy indices, -1 for padding
    token32 = token_of_copy.to(torch.int32)
    row_token = torch.where(row_copy >= 0, token32[row_copy.clamp(min=0).long()], -1).to(torch.int32)
    topo = dataclasses.replace(topo, row_token=row_token)

    h = sdd(x, w1, topo, gelu=True)
    y = dsd(h, w2.view(num_experts * ffn_dim, d_model), topo, version=5)

    rows = y[topo.copy_row.long()].float() * weight_of_copy.float()[:, None]
    out = torch.zeros(num_tokens, d_model, dtype=torch.float32, device=x.device)
    return out.index_add_(0, token_of_copy.long(), rows).to(x.dtype)
