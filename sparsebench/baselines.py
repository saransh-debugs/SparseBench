"""Other ways to do the MoE's second matmul, on the same logical data as ours.

The op: every token copy's FFN activation h [F] times its expert's W [F, D].
All methods start from the same H, built once per token copy and sorted by
expert ("grouped" order, the order build_topology uses inside each expert).
Each method gets H in its own natural layout, and each returns its output
mapped back to grouped order, so one reference checks all of them.

Building a layout is not timed: each system makes its layout during routing.
Only the matmul call is timed.

Optional baselines (torch._grouped_mm, grouped_gemm, stk) may be missing or
unsupported on a given GPU / version. They report why instead of crashing.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sparsebench.routing import DeviceTopology, route_device
from sparsebench.topology import Topology, build_topology

Run = Callable[[], torch.Tensor]
ToGrouped = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Problem:
    num_experts: int
    ffn_dim: int
    d_model: int
    block_size: int
    expert_of_row: torch.Tensor  # [N] int64, expert of grouped row j (N = T*k)
    counts: torch.Tensor  # [E] int64, rows per expert, on the device
    counts_cpu: list[int]  # same, on the host (for methods that need it there)
    h_grouped: torch.Tensor  # [N, F], rows sorted by expert
    w: torch.Tensor  # [E, F, D]
    topo: Topology | DeviceTopology  # our layout (worst-case buffers on CUDA)
    blocks: torch.Tensor  # [nnz, bs, bs], H in our BSR layout
    padded_row: torch.Tensor  # [N] int64, our padded row for grouped row j

    @property
    def w_cat(self) -> torch.Tensor:
        return self.w.view(self.num_experts * self.ffn_dim, self.d_model)

    @property
    def useful_flops(self) -> int:
        return 2 * self.h_grouped.shape[0] * self.ffn_dim * self.d_model


def make_problem(
    expert_idx: torch.Tensor,
    num_experts: int,
    ffn_dim: int,
    d_model: int,
    block_size: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> Problem:
    """On CUDA, our layout uses route_device's worst-case buffers (the real
    no-sync path). On CPU (tests only), the trimmed PyTorch topology."""
    device = expert_idx.device
    flat = expert_idx.reshape(-1).long()
    order = torch.sort(flat, stable=True).indices  # grouped row j -> copy index
    expert_of_row = flat[order]
    counts = torch.bincount(flat, minlength=num_experts)

    g = torch.Generator(device=device).manual_seed(seed)
    # h ~ N(0, 1/F), w ~ N(0, 1): outputs are about N(0, 1).
    h = torch.randn(flat.numel(), ffn_dim, generator=g, device=device, dtype=dtype) / math.sqrt(ffn_dim)
    w = torch.randn(num_experts, ffn_dim, d_model, generator=g, device=device, dtype=dtype)

    topo: Topology | DeviceTopology
    if device.type == "cuda":
        topo = route_device(expert_idx, num_experts, ffn_dim, block_size)
    else:
        topo = build_topology(expert_idx, num_experts, ffn_dim, block_size)
    padded_row = topo.copy_row.long()[order]

    # Scatter H into padded rows, then cut it into bs x bs blocks. Row block r,
    # block j holds H's padded rows of r and expert-local columns j: exactly
    # the block our layout says multiplies W rows (expert * F/bs + j) * bs.
    num_row_blocks = topo.row_offsets.numel() - 1
    ffn_blocks = ffn_dim // block_size
    h_padded = torch.zeros(num_row_blocks * block_size, ffn_dim, device=device, dtype=dtype)
    h_padded[padded_row] = h
    blocks = (
        h_padded.view(num_row_blocks, block_size, ffn_blocks, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_row_blocks * ffn_blocks, block_size, block_size)
        .contiguous()
    )
    del h_padded
    if blocks.shape[0] != topo.col_indices.numel():
        raise AssertionError("layout is not one block per (row block, expert column)")

    return Problem(
        num_experts, ffn_dim, d_model, block_size, expert_of_row, counts, counts.tolist(),
        h, w, topo, blocks, padded_row,
    )  # fmt: skip


def reference(p: Problem) -> torch.Tensor:
    """fp32 (or fp64 for fp64 inputs) result in grouped order. The ground truth."""
    acc = torch.promote_types(p.h_grouped.dtype, torch.float32)
    out = torch.empty(p.h_grouped.shape[0], p.d_model, device=p.h_grouped.device, dtype=acc)
    start = 0
    for e, n in enumerate(p.counts_cpu):
        if n:
            out[start : start + n] = p.h_grouped[start : start + n].to(acc) @ p.w[e].to(acc)
        start += n
    return out


@dataclass(frozen=True)
class Baseline:
    name: str
    host_sync: bool  # does real use need routing results on the CPU?
    note: str
    setup: Callable[[Problem], tuple[Run, ToGrouped]]
    unavailable: Callable[[], str | None] = field(default=lambda: None)


# ---- ours -------------------------------------------------------------------


def _setup_ours(p: Problem) -> tuple[Run, ToGrouped]:
    from sparsebench.spmm import dsd

    return (lambda: dsd(p.blocks, p.w_cat, p.topo, version=4)), (lambda out: out[p.padded_row])


# ---- torch._grouped_mm: one call for all experts, no padding ----------------


def _grouped_mm_unavailable() -> str | None:
    if not hasattr(torch, "_grouped_mm"):
        return f"torch {torch.__version__} has no torch._grouped_mm"
    return None


def _setup_torch_grouped_mm(p: Problem) -> tuple[Run, ToGrouped]:
    offs = torch.cumsum(p.counts, 0).to(torch.int32)  # end row of each group, on the device
    # Some versions want W column-major per expert. Same values, different
    # memory order; try the plain layout first.
    errors = []
    for w in (p.w, p.w.transpose(1, 2).contiguous().transpose(1, 2)):
        try:
            torch._grouped_mm(p.h_grouped, w, offs=offs)
            torch.cuda.synchronize()
        except RuntimeError as e:
            errors.append(str(e).splitlines()[0])
            continue
        return (lambda w=w: torch._grouped_mm(p.h_grouped, w, offs=offs)), (lambda out: out)
    raise RuntimeError(" / ".join(errors))


# ---- grouped_gemm: MegaBlocks' CUTLASS grouped GEMM -------------------------


def _grouped_gemm_unavailable() -> str | None:
    try:
        import grouped_gemm  # noqa: F401
    except ImportError as e:
        return f"grouped_gemm not installed ({e})"
    return None


def _setup_grouped_gemm(p: Problem) -> tuple[Run, ToGrouped]:
    import grouped_gemm as gg

    batch_sizes = torch.tensor(p.counts_cpu, dtype=torch.int64)  # must be on the CPU
    return (lambda: gg.ops.gmm(p.h_grouped, p.w, batch_sizes)), (lambda out: out)


# ---- stk: MegaBlocks' Triton block-sparse DSD -------------------------------


def _stk_unavailable() -> str | None:
    try:
        import stk  # noqa: F401
    except ImportError as e:
        return f"stk not installed ({e})"
    return None


def _setup_stk(p: Problem) -> tuple[Run, ToGrouped]:
    import stk

    if p.block_size != 128:
        raise RuntimeError("stk only supports 128x128 blocks")
    # stk's Matrix needs its sizes on the host, so it takes the real (trimmed)
    # layout. That's the host sync MegaBlocks does in real use, too.
    topo = p.topo.to_topology() if isinstance(p.topo, DeviceTopology) else p.topo
    nnz = topo.nnz_blocks
    num_block_cols = p.num_experts * (p.ffn_dim // 128)
    counts = (topo.row_offsets[1:] - topo.row_offsets[:-1]).long()
    row_indices = torch.repeat_interleave(torch.arange(topo.num_row_blocks, device=counts.device), counts)
    column_indices = topo.col_indices.long()
    # Transposed view of the layout, built the way MegaBlocks builds it.
    gather = torch.sort(column_indices, stable=True).indices
    column_indices_t = row_indices[gather]
    offsets_t = torch.zeros(num_block_cols + 1, dtype=torch.int32, device=counts.device)
    offsets_t[1:] = torch.cumsum(torch.bincount(column_indices, minlength=num_block_cols), 0)
    # MegaBlocks stores block indices as int16 when they fit.
    idx = torch.int16 if max(topo.num_row_blocks, num_block_cols) < 2**15 else torch.int32
    matrix = stk.Matrix(
        (topo.num_rows, p.num_experts * p.ffn_dim),
        p.blocks[:nnz],
        row_indices.to(idx),
        column_indices.to(idx),
        topo.row_offsets.to(torch.int32),
        column_indices_t.to(idx),
        offsets_t,
        gather.to(torch.int32),
    )
    return (lambda: stk.ops.dsd(matrix, p.w_cat)), (lambda out: out[p.padded_row])


# ---- simple ones: per-expert cuBLAS loop, padded bmm ------------------------


def _setup_loop(p: Problem) -> tuple[Run, ToGrouped]:
    def run() -> torch.Tensor:
        out = torch.empty(p.h_grouped.shape[0], p.d_model, device=p.h_grouped.device, dtype=p.h_grouped.dtype)
        start = 0
        for e, n in enumerate(p.counts_cpu):
            if n:
                torch.mm(p.h_grouped[start : start + n], p.w[e], out=out[start : start + n])
            start += n
        return out

    return run, (lambda out: out)


def _setup_padded_bmm(p: Problem) -> tuple[Run, ToGrouped]:
    # GShard / Switch style: every expert padded to the busiest one's size.
    capacity = max(p.counts_cpu)
    starts = torch.cumsum(p.counts, 0) - p.counts
    rank = torch.arange(p.expert_of_row.numel(), device=p.counts.device) - starts[p.expert_of_row]
    x = p.h_grouped.new_zeros(p.num_experts, capacity, p.ffn_dim)
    x[p.expert_of_row, rank] = p.h_grouped
    return (lambda: torch.bmm(x, p.w)), (lambda out: out[p.expert_of_row, rank])


BASELINES = [
    Baseline("ours (v4)", False, "block-sparse, worst-case buffers", _setup_ours),
    Baseline("torch._grouped_mm", False, "grouped GEMM, no padding", _setup_torch_grouped_mm, _grouped_mm_unavailable),
    Baseline("grouped_gemm (CUTLASS)", True, "MegaBlocks' grouped GEMM", _setup_grouped_gemm, _grouped_gemm_unavailable),
    Baseline("stk (MegaBlocks Triton DSD)", True, "block-sparse, needs nnz on host", _setup_stk, _stk_unavailable),
    Baseline("per-expert cuBLAS loop", True, "one torch.mm per expert", _setup_loop),
    Baseline("padded bmm", True, "pad every expert to the busiest", _setup_padded_bmm),
]
