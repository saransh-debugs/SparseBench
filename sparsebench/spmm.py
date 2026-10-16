"""Block-sparse matmul kernels."""

import torch

from sparsebench.routing import DeviceTopology
from sparsebench.topology import Topology


def dsd(
    blocks: torch.Tensor, w2_cat: torch.Tensor, topo: Topology | DeviceTopology, version: int = 1
) -> torch.Tensor:
    """out = H @ w2_cat, with H block-sparse in topo's layout. Same math as reference.dsd_reference.

    blocks [nnz, bs, bs] bf16, w2_cat [E*F, D] bf16 -> [num_row_blocks*bs, D] bf16.
    With a DeviceTopology, everything is worst-case sized: rows past the real
    end come out as zeros, and no host sync happens.
    """
    from sparsebench import _C

    bs = topo.block_size
    if blocks.dim() != 3 or blocks.shape[1:] != (bs, bs):
        raise ValueError(f"blocks must be [nnz, {bs}, {bs}] for this topology, got {tuple(blocks.shape)}")
    return _C.dsd_cuda(blocks, w2_cat, topo.row_offsets, topo.col_indices, version)
