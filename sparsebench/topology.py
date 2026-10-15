"""Routing decisions -> block-sparse layout.

Each token picks k experts, so there are T*k "copies" of tokens. We sort the
copies by expert and pad each expert's group up to a multiple of block_size.
That gives P padded rows, split into row blocks that each belong to one expert.

The first MoE matmul's output (and the second's input) is a [P, E*F] matrix.
Row block r only touches its own expert's F columns, so only those blocks are
non-zero. That's the block-sparse layout, stored as BSR:
  row_offsets[r] .. row_offsets[r+1]  ->  indices into col_indices
  col_indices[i]                      ->  column block of non-zero block i

Why pad to block_size and not to a fixed capacity per expert: capacity padding
makes every expert as big as the busiest one, so skewed routing wastes most of
the FLOPs (or drops tokens). Block padding wastes under one block per expert.

This is the plain PyTorch version. The CUDA routing kernel must match it exactly.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Topology:
    block_size: int
    num_experts: int
    ffn_blocks: int  # F // block_size: column blocks per expert

    tokens_per_expert: torch.Tensor  # [E] int32, real copies per expert
    padded_offsets: torch.Tensor  # [E+1] int32, first padded row of each expert
    row_token: torch.Tensor  # [P] int32, source token of each row, -1 = padding
    copy_row: torch.Tensor  # [T*k] int32, padded row of copy (t, j) at t*k + j

    row_block_expert: torch.Tensor  # [P / block_size] int32
    row_offsets: torch.Tensor  # [P / block_size + 1] int32
    col_indices: torch.Tensor  # [nnz_blocks] int32

    @property
    def num_rows(self) -> int:
        return self.row_token.numel()

    @property
    def num_row_blocks(self) -> int:
        return self.row_block_expert.numel()

    @property
    def nnz_blocks(self) -> int:
        return self.col_indices.numel()


def build_topology(
    expert_idx: torch.Tensor, num_experts: int, ffn_dim: int, block_size: int
) -> Topology:
    """expert_idx: [T, k] integer tensor, the experts each token was routed to."""
    if expert_idx.dim() != 2 or expert_idx.shape[0] == 0 or expert_idx.shape[1] == 0:
        raise ValueError(f"expert_idx must be [T, k] with T, k > 0, got {tuple(expert_idx.shape)}")
    if expert_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"expert_idx must be int32 or int64, got {expert_idx.dtype}")
    if num_experts <= 0 or block_size <= 0:
        raise ValueError("num_experts and block_size must be positive")
    if ffn_dim <= 0 or ffn_dim % block_size != 0:
        raise ValueError(f"ffn_dim ({ffn_dim}) must be a positive multiple of block_size ({block_size})")
    lo, hi = int(expert_idx.min()), int(expert_idx.max())
    if lo < 0 or hi >= num_experts:
        raise ValueError(f"expert ids must be in [0, {num_experts}), got range [{lo}, {hi}]")

    device = expert_idx.device
    num_tokens, k = expert_idx.shape
    flat = expert_idx.reshape(-1).long()  # copy c = t*k + j

    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    padded_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    padded_offsets[1:] = torch.cumsum(padded, 0)
    num_rows = int(padded_offsets[-1])
    if num_rows * ffn_dim >= 2**31:
        # The sparse matrix holds P*F values, and kernels index it with int32.
        # Fail here instead of silently wrapping around there.
        raise ValueError("problem too large for int32 indexing")

    # Stable sort: copies of the same expert keep token order. The CUDA kernel
    # must produce this exact order, so tests can compare bit for bit.
    order = torch.sort(flat, stable=True).indices
    sorted_expert = flat[order]
    count_offsets = torch.cumsum(counts, 0) - counts
    rank_in_expert = torch.arange(flat.numel(), device=device) - count_offsets[sorted_expert]
    dest_row = padded_offsets[sorted_expert] + rank_in_expert

    copy_row = torch.empty_like(flat)
    copy_row[order] = dest_row
    row_token = torch.full((num_rows,), -1, dtype=torch.long, device=device)
    row_token[dest_row] = order // k

    ffn_blocks = ffn_dim // block_size
    row_block_expert = torch.repeat_interleave(
        torch.arange(num_experts, device=device), padded // block_size
    )
    # Every row block has exactly ffn_blocks non-zero blocks: its expert's columns.
    row_offsets = torch.arange(row_block_expert.numel() + 1, device=device) * ffn_blocks
    col_indices = (
        row_block_expert[:, None] * ffn_blocks + torch.arange(ffn_blocks, device=device)
    ).reshape(-1)

    i32 = torch.int32
    return Topology(
        block_size=block_size,
        num_experts=num_experts,
        ffn_blocks=ffn_blocks,
        tokens_per_expert=counts.to(i32),
        padded_offsets=padded_offsets.to(i32),
        row_token=row_token.to(i32),
        copy_row=copy_row.to(i32),
        row_block_expert=row_block_expert.to(i32),
        row_offsets=row_offsets.to(i32),
        col_indices=col_indices.to(i32),
    )


def _nnz_row_blocks(topo: Topology) -> torch.Tensor:
    """Row block of each non-zero block, [nnz_blocks]."""
    counts = (topo.row_offsets[1:] - topo.row_offsets[:-1]).long()
    return torch.repeat_interleave(torch.arange(topo.num_row_blocks, device=counts.device), counts)


def bsr_to_dense(blocks: torch.Tensor, topo: Topology) -> torch.Tensor:
    """[nnz, bs, bs] block values -> dense [P, E*F] with zeros elsewhere."""
    bs = topo.block_size
    if blocks.shape != (topo.nnz_blocks, bs, bs):
        raise ValueError(f"expected blocks of shape {(topo.nnz_blocks, bs, bs)}, got {tuple(blocks.shape)}")
    num_col_blocks = topo.num_experts * topo.ffn_blocks
    grid = blocks.new_zeros(topo.num_row_blocks, num_col_blocks, bs, bs)
    grid[_nnz_row_blocks(topo), topo.col_indices.long()] = blocks
    return grid.permute(0, 2, 1, 3).reshape(topo.num_rows, num_col_blocks * bs)


def dense_to_bsr(dense: torch.Tensor, topo: Topology) -> torch.Tensor:
    """Dense [P, E*F] -> [nnz, bs, bs]. Values outside the layout are dropped."""
    bs = topo.block_size
    num_col_blocks = topo.num_experts * topo.ffn_blocks
    if dense.shape != (topo.num_rows, num_col_blocks * bs):
        raise ValueError(f"expected dense of shape {(topo.num_rows, num_col_blocks * bs)}, got {tuple(dense.shape)}")
    grid = dense.reshape(topo.num_row_blocks, bs, num_col_blocks, bs).permute(0, 2, 1, 3)
    return grid[_nnz_row_blocks(topo), topo.col_indices.long()].contiguous()
