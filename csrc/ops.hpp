#pragma once

#include <torch/extension.h>

namespace sparsebench {

// Returns an int32 tensor of length n. Every element holds __CUDA_ARCH__
// of the GPU code that actually ran (800 on A100, 900 on H100).
// Proves the build shipped code for this GPU, not just PTX that gets
// recompiled at load time.
torch::Tensor arch_probe(int64_t n);

// GPU version of sparsebench/topology.py:build_topology. Never syncs.
// Outputs are sized for the worst case; the real padded row count is
// padded_offsets[E]. Returns, in order:
//   tokens_per_expert [E], padded_offsets [E+1], row_token [max_rows],
//   copy_row [T*k], row_block_expert [max_row_blocks],
//   row_offsets [max_row_blocks+1], col_indices [max_row_blocks*ffn_blocks],
//   error [1] (non-zero if any expert id was out of range)
std::vector<torch::Tensor> build_topology_cuda(const torch::Tensor& expert_idx, int64_t num_experts,
                                               int64_t ffn_dim, int64_t block_size);

// out [num_row_blocks*bs, D] = BSR(blocks, row_offsets, col_indices) @ w.
// blocks [nnz, bs, bs] bf16, w [E*F, D] bf16, bf16 out with fp32 accumulation.
// version picks the kernel (1 = baseline). Row blocks with no non-zero
// blocks come out as zeros.
torch::Tensor dsd_cuda(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                       const torch::Tensor& col_indices, int64_t version);

}  // namespace sparsebench
