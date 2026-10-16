// Checks and dispatch for every DSD kernel version. Checks run once here so
// each version's launcher can assume clean input.

#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "dsd.hpp"
#include "ops.hpp"

namespace sparsebench {
namespace {

void check_index_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 1 && t.is_contiguous(), name,
              " must be a contiguous 1-D int32 CUDA tensor");
}

void check_aligned(const torch::Tensor& t, const char* name) {
  // Kernels load 16 bytes at a time.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

}  // namespace

torch::Tensor dsd_cuda(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                       const torch::Tensor& col_indices, int64_t version) {
  TORCH_CHECK(blocks.is_cuda() && w.is_cuda(), "blocks and w must be CUDA tensors");
  TORCH_CHECK(blocks.device() == w.device() && row_offsets.device() == w.device() &&
                  col_indices.device() == w.device(),
              "all inputs must be on the same GPU");
  TORCH_CHECK(blocks.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
              "blocks and w must be bfloat16");
  TORCH_CHECK(blocks.is_contiguous() && w.is_contiguous(), "blocks and w must be contiguous");
  TORCH_CHECK(blocks.dim() == 3 && blocks.size(1) == blocks.size(2), "blocks must be [nnz, bs, bs], got ",
              blocks.sizes());
  TORCH_CHECK(w.dim() == 2, "w must be [E*F, D], got ", w.sizes());
  check_index_tensor(row_offsets, "row_offsets");
  check_index_tensor(col_indices, "col_indices");

  const int64_t bs = blocks.size(1);
  const int64_t d_model = w.size(1);
  const int64_t num_row_blocks = row_offsets.numel() - 1;
  TORCH_CHECK(bs == 64 || bs == 128, "block size must be 64 or 128, got ", bs);
  TORCH_CHECK(d_model > 0 && d_model % kDsdTileN == 0, "D (", d_model, ") must be a positive multiple of ",
              kDsdTileN);
  TORCH_CHECK(d_model / kDsdTileN <= 65535, "D too large for grid y");
  TORCH_CHECK(w.size(0) % bs == 0, "w rows (", w.size(0), ") must be a multiple of the block size (", bs, ")");
  TORCH_CHECK(num_row_blocks >= 0, "row_offsets must have at least one entry");
  // blocks and col_indices are both indexed by non-zero position, so they
  // must be the same length. Their values (which W rows each block uses)
  // can't be checked without a GPU sync; topology.py / routing.cu produce them.
  TORCH_CHECK(blocks.size(0) == col_indices.numel(), "blocks has ", blocks.size(0),
              " blocks but col_indices has ", col_indices.numel());
  TORCH_CHECK(num_row_blocks * bs <= std::numeric_limits<int32_t>::max(), "too many row blocks");
  check_aligned(blocks, "blocks");
  check_aligned(w, "w");

  const at::cuda::OptionalCUDAGuard guard(w.device());
  auto out = torch::empty({num_row_blocks * bs, d_model}, w.options());
  if (num_row_blocks == 0) return out;

  switch (version) {
    case 1: dsd_v1_launch(blocks, w, row_offsets, col_indices, out); break;
    case 2: dsd_v2_launch(blocks, w, row_offsets, col_indices, out, 3); break;
    // Not a real version: v2 with 2 stages, so the benchmark can show why v2 uses 3.
    case 21: dsd_v2_launch(blocks, w, row_offsets, col_indices, out, 2); break;
    case 3: dsd_v3_launch(blocks, w, row_offsets, col_indices, out, 3); break;
    case 31: dsd_v3_launch(blocks, w, row_offsets, col_indices, out, 2); break;  // benchmark only
    default: TORCH_CHECK(false, "unknown dsd version ", version);
  }
  return out;
}

}  // namespace sparsebench
