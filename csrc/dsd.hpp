#pragma once
// Internal: one launcher per DSD kernel version. Inputs are already checked
// by dsd_cuda in dsd.cpp, and `out` is allocated there.

#include <torch/extension.h>

namespace sparsebench {

constexpr int kDsdTileN = 128;  // every version tiles D in slices of 128 columns

void dsd_v1_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out);

}  // namespace sparsebench
