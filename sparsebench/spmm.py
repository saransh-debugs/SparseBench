"""Block-sparse matmul kernels."""

import torch

from sparsebench.routing import DeviceTopology
from sparsebench.topology import Topology


def dsd(
    blocks: torch.Tensor, w2_cat: torch.Tensor, topo: Topology | DeviceTopology, version: int = 1
) -> torch.Tensor:
    """out = H @ w2_cat, with H block-sparse in topo's layout. Same math as reference.dsd_reference.

    blocks [nnz, bs, bs] bf16, w2_cat [E*F, D] bf16 -> [num_row_blocks*bs, D] bf16.
    With a DeviceTopology, everything is worst-case sized and no host sync
    happens. Rows of empty row blocks (past the real end) come out as zeros in
    v1-v3. From v4 on they are left unwritten: skipping them is the point, and
    nothing downstream reads them.
    """
    from sparsebench import _C

    bs = topo.block_size
    if blocks.dim() != 3 or blocks.shape[1:] != (bs, bs):
        raise ValueError(f"blocks must be [nnz, {bs}, {bs}] for this topology, got {tuple(blocks.shape)}")
    return _C.dsd_cuda(blocks, w2_cat, topo.row_offsets, topo.col_indices, version)


def sdd(x: torch.Tensor, w1: torch.Tensor, topo: Topology | DeviceTopology, gelu: bool = True) -> torch.Tensor:
    """H = (gelu)(group_tokens(x) @ w1[expert]), kept only on topo's blocks.
    Same math as reference.sdd_reference (+ gelu), without the dense product.

    x [T, D] bf16, w1 [E, D, F] bf16 -> blocks [nnz, 128, 128] bf16 in topo's
    BSR order, ready for dsd(). Rows are gathered from x inside the kernel.
    With a DeviceTopology, nothing syncs; blocks of dead row blocks are left
    unwritten. Needs block size 128, D % 64 == 0 and F % 256 == 0.
    """
    from sparsebench import _C

    if topo.block_size != 128:
        raise ValueError(f"sdd needs block size 128, got {topo.block_size}")
    return _C.sdd_cuda(
        x, w1, topo.row_token, topo.row_block_expert, topo.row_offsets, topo.col_indices.numel(), gelu
    )
