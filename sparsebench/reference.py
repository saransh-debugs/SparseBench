"""Plain PyTorch MoE layer. Ground truth for every kernel.

Expert e computes  y = gelu(x @ w1[e]) @ w2[e].
Each token goes to k experts and gets the weighted sum of their outputs.

Two versions on purpose:
- moe_reference: loop over experts. Obviously correct, nothing sparse in it.
- moe_blocksparse_reference: the same math through the block-sparse layout,
  with the two matmuls split out (sdd_reference, dsd_reference). Our CUDA
  kernels replace those two functions, so they get tested one at a time.
"""

import torch
import torch.nn.functional as F

from sparsebench.topology import Topology, build_topology, bsr_to_dense, dense_to_bsr


def route(logits: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing, Mixtral style: softmax over the k picked logits only.

    logits: [T, E]. Returns (expert_idx [T, k] int64, expert_weight [T, k]).
    """
    if logits.dim() != 2 or not 0 < k <= logits.shape[1]:
        raise ValueError(f"need [T, E] logits with 0 < k <= E, got {tuple(logits.shape)} and k={k}")
    top_logits, expert_idx = torch.topk(logits, k, dim=-1)
    # Softmax in fp32: in bf16, close logits round to the same value.
    weights = torch.softmax(top_logits.float(), dim=-1).to(logits.dtype)
    return expert_idx, weights


def _check_shapes(
    x: torch.Tensor, expert_idx: torch.Tensor, expert_weight: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> None:
    if x.dim() != 2:
        raise ValueError(f"x must be [T, D], got {tuple(x.shape)}")
    num_tokens, d_model = x.shape
    if w1.dim() != 3 or w1.shape[1] != d_model:
        raise ValueError(f"w1 must be [E, {d_model}, F], got {tuple(w1.shape)}")
    num_experts, _, ffn_dim = w1.shape
    if w2.shape != (num_experts, ffn_dim, d_model):
        raise ValueError(f"w2 must be {(num_experts, ffn_dim, d_model)}, got {tuple(w2.shape)}")
    if expert_idx.dim() != 2 or expert_idx.shape[0] != num_tokens:
        raise ValueError(f"expert_idx must be [{num_tokens}, k], got {tuple(expert_idx.shape)}")
    if expert_weight.shape != expert_idx.shape:
        raise ValueError(f"expert_weight must match expert_idx shape {tuple(expert_idx.shape)}")


def moe_reference(
    x: torch.Tensor, expert_idx: torch.Tensor, expert_weight: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """x [T, D], expert_idx/expert_weight [T, k], w1 [E, D, F], w2 [E, F, D] -> [T, D]."""
    _check_shapes(x, expert_idx, expert_weight, w1, w2)
    out = torch.zeros_like(x)
    for e in range(w1.shape[0]):
        token, slot = torch.nonzero(expert_idx == e, as_tuple=True)
        if token.numel() == 0:
            continue
        y = F.gelu(x[token] @ w1[e]) @ w2[e]
        out.index_add_(0, token, y * expert_weight[token, slot, None])
    return out


def sdd_reference(xg: torch.Tensor, w1_cat: torch.Tensor, topo: Topology) -> torch.Tensor:
    """Sparse output = dense @ dense, kept only on the layout's blocks.

    xg [P, D] grouped tokens, w1_cat [D, E*F] -> block values [nnz, bs, bs].
    Computes the full dense product, which is the waste the kernel avoids.
    """
    return dense_to_bsr(xg @ w1_cat, topo)


def dsd_reference(blocks: torch.Tensor, w2_cat: torch.Tensor, topo: Topology) -> torch.Tensor:
    """Dense output = sparse @ dense.

    blocks [nnz, bs, bs] (layout from topo), w2_cat [E*F, D] -> [P, D].
    """
    return bsr_to_dense(blocks, topo) @ w2_cat


def group_tokens(x: torch.Tensor, topo: Topology) -> torch.Tensor:
    """[T, D] -> [P, D] in expert order. Padding rows are zero."""
    row_token = topo.row_token.long()
    xg = x[row_token.clamp(min=0)]
    xg[row_token < 0] = 0
    return xg


def ungroup_combine(y_rows: torch.Tensor, expert_weight: torch.Tensor, topo: Topology) -> torch.Tensor:
    """[P, D] -> [T, D]: pick each token's k rows and take the weighted sum."""
    num_tokens, k = expert_weight.shape
    y = y_rows[topo.copy_row.long()].view(num_tokens, k, -1)
    return (y * expert_weight[..., None]).sum(dim=1)


def moe_blocksparse_reference(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    expert_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Same result as moe_reference, computed through the block-sparse layout."""
    _check_shapes(x, expert_idx, expert_weight, w1, w2)
    num_experts, d_model, ffn_dim = w1.shape
    topo = build_topology(expert_idx, num_experts, ffn_dim, block_size)

    w1_cat = w1.permute(1, 0, 2).reshape(d_model, num_experts * ffn_dim)
    w2_cat = w2.reshape(num_experts * ffn_dim, d_model)

    h = sdd_reference(group_tokens(x, topo), w1_cat, topo)
    y_rows = dsd_reference(F.gelu(h), w2_cat, topo)
    return ungroup_combine(y_rows, expert_weight, topo)
