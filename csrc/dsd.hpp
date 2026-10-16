#pragma once
// Internal: one launcher per DSD kernel version. Inputs are already checked
// by dsd_cuda in dsd.cpp, and `out` is allocated there.

#include <torch/extension.h>

namespace sparsebench {

constexpr int kDsdTileN = 128;  // every version tiles D in slices of 128 columns

void dsd_v1_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out);

// stages: shared-memory ring size. v2 uses 3; 2 is only for the benchmark.
void dsd_v2_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out, int stages);

void dsd_v3_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out, int stages);

// Leaves rows of empty row blocks unwritten.
void dsd_v4_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out, int stages);

}  // namespace sparsebench
