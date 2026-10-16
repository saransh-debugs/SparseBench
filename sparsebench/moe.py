"""A full MoE layer forward pass on our kernels, with no host sync anywhere.

    routing (routing.cu) -> SDD + GELU (sdd.cu) -> DSD v5 (dsd_v5.cu) -> combine

Every buffer is sized for the worst case on the host, so the CPU never waits
for the GPU to learn how many rows routing produced. Same math as
reference.moe_reference.
"""

import torch

from sparsebench.routing import route_device
from sparsebench.spmm import dsd, sdd

BLOCK_SIZE = 128


def moe_forward(
    x: torch.Tensor, expert_idx: torch.Tensor, expert_weight: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """x [T, D], expert_idx / expert_weight [T, k], w1 [E, D, F], w2 [E, F, D] -> [T, D].
    bf16 weights and activations; fp32 accumulation inside the kernels."""
    num_experts, d_model, ffn_dim = w1.shape
    if w2.shape != (num_experts, ffn_dim, d_model):
        raise ValueError(f"w2 must be {(num_experts, ffn_dim, d_model)}, got {tuple(w2.shape)}")
    if x.shape[1] != d_model or expert_idx.shape != expert_weight.shape or expert_idx.shape[0] != x.shape[0]:
        raise ValueError("x must be [T, D] and expert_idx / expert_weight [T, k]")

    topo = route_device(expert_idx, num_experts, ffn_dim, BLOCK_SIZE)
    h = sdd(x, w1, topo, gelu=True)
    y = dsd(h, w2.view(num_experts * ffn_dim, d_model), topo, version=5)

    # Each token's k rows, weighted. Indexing with GPU tensors: no sync.
    num_tokens, k = expert_idx.shape
    rows = y[topo.copy_row.long()].view(num_tokens, k, d_model)
    return (rows.float() * expert_weight.float()[..., None]).sum(dim=1).to(x.dtype)
