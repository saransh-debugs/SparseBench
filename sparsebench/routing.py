"""GPU routing: same result as topology.build_topology, without host syncs.

route_device() never waits on the GPU. Its buffers are sized for the worst
case, and the real padded row count lives in padded_offsets[E] on the GPU.
Kernels read it from there. Only to_topology() syncs, to trim the buffers
for tests and debugging.
"""

from dataclasses import dataclass

import torch

from sparsebench.topology import Topology


@dataclass(frozen=True)
class DeviceTopology:
    block_size: int
    num_experts: int
    ffn_blocks: int

    tokens_per_expert: torch.Tensor  # [E]
    padded_offsets: torch.Tensor  # [E+1], padded_offsets[E] = real row count
    row_token: torch.Tensor  # [max_rows], -1 past the real end too
    copy_row: torch.Tensor  # [T*k]
    row_block_expert: torch.Tensor  # [max_row_blocks], -1 past the real end
    row_offsets: torch.Tensor  # [max_row_blocks + 1]
    col_indices: torch.Tensor  # [max_row_blocks * ffn_blocks], -1 past the real end
    error: torch.Tensor  # [1], non-zero if any expert id was out of range

    def check(self) -> None:
        """Syncs. Raises if routing had an out-of-range expert id."""
        if int(self.error.item()) != 0:
            raise ValueError(f"expert_idx has ids outside [0, {self.num_experts})")

    def to_topology(self, check: bool = True) -> Topology:
        """Syncs, then trims the worst-case buffers to the real sizes."""
        if check:
            self.check()
        num_rows = int(self.padded_offsets[-1].item())
        num_row_blocks = num_rows // self.block_size
        return Topology(
            block_size=self.block_size,
            num_experts=self.num_experts,
            ffn_blocks=self.ffn_blocks,
            tokens_per_expert=self.tokens_per_expert,
            padded_offsets=self.padded_offsets,
            row_token=self.row_token[:num_rows],
            copy_row=self.copy_row,
            row_block_expert=self.row_block_expert[:num_row_blocks],
            row_offsets=self.row_offsets[: num_row_blocks + 1],
            col_indices=self.col_indices[: num_row_blocks * self.ffn_blocks],
        )


def route_device(expert_idx: torch.Tensor, num_experts: int, ffn_dim: int, block_size: int) -> DeviceTopology:
    """expert_idx: [T, k] int32/int64 CUDA tensor. No host sync."""
    from sparsebench import _C

    out = _C.build_topology_cuda(expert_idx, num_experts, ffn_dim, block_size)
    return DeviceTopology(block_size, num_experts, ffn_dim // block_size, *out)


def build_topology_cuda(
    expert_idx: torch.Tensor, num_experts: int, ffn_dim: int, block_size: int, check: bool = True
) -> Topology:
    """Drop-in for topology.build_topology on CUDA tensors. Syncs once at the end."""
    return route_device(expert_idx, num_experts, ffn_dim, block_size).to_topology(check)
