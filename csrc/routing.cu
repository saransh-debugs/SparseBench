// Routing -> block-sparse layout on the GPU. Same output as
// sparsebench/topology.py:build_topology, bit for bit.
//
// A "copy" is one (token, choice) pair, numbered c = t*k + j. We sort copies by
// expert with a counting sort, keeping copy order inside each expert:
//   1. count_kernel    per thread block: how many copies of each expert
//   2. offsets_kernel  per expert: copies before each thread block, padded offsets
//   3. scatter_kernel  each copy's final row
//   4. layout_kernel   BSR arrays for the matmul kernels
//
// No host sync anywhere. The padded row count P is only known on the GPU, so
// outputs are sized for the worst case and P stays in padded_offsets[E].
// A sync per MoE layer would stall the CPU until the GPU drains, leaving the
// GPU idle while the CPU queues the next kernels.

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "ops.hpp"

namespace sparsebench {
namespace {

constexpr int kChunk = 256;  // copies per thread block in kernels 1 and 3
constexpr int kWarps = kChunk / 32;
constexpr int kMaxExperts = 1024;  // kernel 2 runs one thread per expert in one block
constexpr unsigned kFullMask = 0xffffffffu;

template <typename index_t>
__global__ void count_kernel(const index_t* __restrict__ expert_idx, int32_t n, int32_t num_experts,
                             int32_t* __restrict__ block_counts, int32_t* __restrict__ error) {
  extern __shared__ int32_t hist[];  // [num_experts]
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) hist[e] = 0;
  __syncthreads();

  const int64_t c = static_cast<int64_t>(blockIdx.x) * kChunk + threadIdx.x;
  if (c < n) {
    const index_t e = expert_idx[c];
    if (e >= 0 && e < num_experts) {
      // Order doesn't matter for a count, so shared-memory atomics are fine here.
      atomicAdd(&hist[e], 1);
    } else {
      // Bad id: skip the copy so nothing writes out of bounds. The host
      // checks this flag only when asked, so there is no sync by default.
      atomicOr(error, 1);
    }
  }
  __syncthreads();

  int32_t* out = block_counts + static_cast<int64_t>(blockIdx.x) * num_experts;
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) out[e] = hist[e];
}

// One block, one thread per expert (blockDim rounded up to a multiple of 32).
// Turns block_counts into block_base in place: copies of expert e in all
// earlier thread blocks.
//
// The walk down each column is serial: num_blocks steps per thread. That's
// ~800 steps at 200k copies, a few microseconds. A single-pass scan with
// decoupled look-back would remove it, but it isn't the bottleneck here.
__global__ void offsets_kernel(int32_t* __restrict__ block_counts, int32_t num_blocks, int32_t num_experts,
                               int32_t block_size, int32_t* __restrict__ tokens_per_expert,
                               int32_t* __restrict__ padded_offsets) {
  __shared__ int32_t warp_sums[32];
  const int e = threadIdx.x;
  const int lane = e & 31;
  const int warp = e >> 5;

  int32_t total = 0;
  if (e < num_experts) {
    for (int32_t b = 0; b < num_blocks; ++b) {
      int32_t* slot = block_counts + static_cast<int64_t>(b) * num_experts + e;
      const int32_t count = *slot;
      *slot = total;
      total += count;
    }
    tokens_per_expert[e] = total;
  }

  // Inclusive prefix sum of padded sizes across experts.
  // Step 1, inside each warp: __shfl_up_sync reads a register from the lane
  // d below. No shared memory, no __syncthreads, log2(32) = 5 steps.
  int32_t x = (e < num_experts) ? (total + block_size - 1) / block_size * block_size : 0;
  for (int d = 1; d < 32; d <<= 1) {
    const int32_t below = __shfl_up_sync(kFullMask, x, d);
    if (lane >= d) x += below;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();

  // Step 2: warp 0 scans the per-warp totals the same way.
  if (warp == 0) {
    const int num_warps = blockDim.x / 32;
    int32_t s = (lane < num_warps) ? warp_sums[lane] : 0;
    for (int d = 1; d < 32; d <<= 1) {
      const int32_t below = __shfl_up_sync(kFullMask, s, d);
      if (lane >= d) s += below;
    }
    warp_sums[lane] = s;  // each lane writes only its own slot: no race
  }
  __syncthreads();

  if (e < num_experts) padded_offsets[e + 1] = x + (warp > 0 ? warp_sums[warp - 1] : 0);
  if (e == 0) padded_offsets[0] = 0;
}

// Must use the same chunks as count_kernel so block_base lines up.
template <typename index_t>
__global__ void scatter_kernel(const index_t* __restrict__ expert_idx, int32_t n, int32_t num_experts, int32_t k,
                               const int32_t* __restrict__ block_base, const int32_t* __restrict__ padded_offsets,
                               int32_t* __restrict__ copy_row, int32_t* __restrict__ row_token) {
  extern __shared__ int32_t warp_counts[];  // [kWarps][num_experts]
  for (int i = threadIdx.x; i < kWarps * num_experts; i += blockDim.x) warp_counts[i] = 0;
  __syncthreads();

  const int64_t c = static_cast<int64_t>(blockIdx.x) * kChunk + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  int32_t e = -1;
  if (c < n) {
    const index_t v = expert_idx[c];
    if (v >= 0 && v < num_experts) e = static_cast<int32_t>(v);
  }
  // Which lanes hold a real copy. The last warp of the last block can be
  // partly past n, and bad ids are skipped. All 32 lanes reach this line.
  const bool valid = e >= 0;
  const unsigned active = __ballot_sync(kFullMask, valid);

  int32_t rank = 0;
  if (valid) {
    // Bitmask of active lanes with the same expert as this lane.
    // Rank = how many of them sit at lower lanes, i.e. come earlier in copy
    // order. That's a stable order with no atomics.
    //
    // Why not atomicAdd on a per-expert counter: the order would change every
    // run, and with skewed routing all threads hit one counter and serialize.
    const unsigned peers = __match_any_sync(active, e);
    const unsigned lower = peers & ((1u << lane) - 1u);
    rank = __popc(lower);
    if (lower == 0) warp_counts[warp * num_experts + e] = __popc(peers);  // one writer per (warp, e)
  }
  __syncthreads();

  if (valid) {
    int32_t before = 0;
    for (int w = 0; w < warp; ++w) before += warp_counts[w * num_experts + e];
    const int32_t row = padded_offsets[e] + block_base[static_cast<int64_t>(blockIdx.x) * num_experts + e] +
                        before + rank;
    copy_row[c] = row;
    row_token[row] = static_cast<int32_t>(c / k);
  } else if (c < n) {
    copy_row[c] = -1;
  }
}

// One thread per BSR block slot, up to the worst case. Every real row block
// holds exactly ffn_blocks blocks (its expert's columns). Slots past the real
// end get -1 so the matmul kernels can skip them.
__global__ void layout_kernel(const int32_t* __restrict__ padded_offsets, int32_t num_experts, int32_t block_size,
                              int32_t ffn_blocks, int32_t max_row_blocks, int32_t* __restrict__ row_block_expert,
                              int32_t* __restrict__ row_offsets, int32_t* __restrict__ col_indices) {
  // 64-bit: the last block's spare threads can run past the int32 range.
  const int64_t i64 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i64 > static_cast<int64_t>(max_row_blocks) * ffn_blocks) return;
  const auto i = static_cast<int32_t>(i64);  // host checked it fits
  const int32_t num_row_blocks = padded_offsets[num_experts] / block_size;

  if (i <= max_row_blocks) row_offsets[i] = min(i, num_row_blocks) * ffn_blocks;
  if (i == max_row_blocks * ffn_blocks) return;

  const int32_t r = i / ffn_blocks;
  const int32_t j = i % ffn_blocks;
  int32_t expert = -1;
  if (r < num_row_blocks) {
    // Last expert whose range starts at or before this row. Empty experts
    // share a start with the next one, so "last" skips past them.
    const int32_t row = r * block_size;
    int32_t lo = 0, hi = num_experts - 1;
    while (lo < hi) {
      const int32_t mid = (lo + hi + 1) / 2;
      if (padded_offsets[mid] <= row) lo = mid; else hi = mid - 1;
    }
    expert = lo;
  }
  col_indices[i] = expert < 0 ? -1 : expert * ffn_blocks + j;
  if (j == 0) row_block_expert[r] = expert;
}

}  // namespace

std::vector<torch::Tensor> build_topology_cuda(const torch::Tensor& expert_idx, int64_t num_experts,
                                               int64_t ffn_dim, int64_t block_size) {
  TORCH_CHECK(expert_idx.is_cuda(), "expert_idx must be a CUDA tensor");
  TORCH_CHECK(expert_idx.dim() == 2 && expert_idx.size(0) > 0 && expert_idx.size(1) > 0,
              "expert_idx must be [T, k] with T, k > 0, got ", expert_idx.sizes());
  TORCH_CHECK(expert_idx.scalar_type() == torch::kInt32 || expert_idx.scalar_type() == torch::kInt64,
              "expert_idx must be int32 or int64, got ", expert_idx.scalar_type());
  TORCH_CHECK(expert_idx.is_contiguous(), "expert_idx must be contiguous");
  TORCH_CHECK(num_experts > 0 && num_experts <= kMaxExperts, "num_experts must be in [1, ", kMaxExperts,
              "], got ", num_experts);
  constexpr int64_t kI32Max = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(block_size > 0 && block_size <= kI32Max, "block_size must be in [1, 2^31)");
  TORCH_CHECK(ffn_dim > 0 && ffn_dim % block_size == 0, "ffn_dim (", ffn_dim,
              ") must be a positive multiple of block_size (", block_size, ")");

  const int64_t n = expert_idx.numel();
  // Worst case: every expert wastes block_size - 1 rows of padding.
  // Fits in int64: n < 2^63, num_experts <= 2^10, block_size < 2^31.
  const int64_t max_row_blocks = (n + num_experts * (block_size - 1)) / block_size;
  const int64_t max_rows = max_row_blocks * block_size;
  const int64_t ffn_blocks = ffn_dim / block_size;
  // Kernels index with int32. Same limit as the Python version, applied to
  // the worst case because the real P isn't known on the host. Divide
  // instead of multiply so the check itself can't overflow.
  TORCH_CHECK(max_rows <= kI32Max / ffn_dim, "problem too large for int32 indexing");
  // Implied by the line above (max_row_blocks * ffn_blocks <= max_rows * ffn_dim),
  // spelled out because layout_kernel launches one extra thread past it.
  TORCH_CHECK(max_row_blocks * ffn_blocks < kI32Max, "problem too large for int32 indexing");

  const at::cuda::OptionalCUDAGuard guard(expert_idx.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto opts = expert_idx.options().dtype(torch::kInt32);
  const auto n32 = static_cast<int32_t>(n);
  const auto num_experts32 = static_cast<int32_t>(num_experts);
  const auto block_size32 = static_cast<int32_t>(block_size);
  const auto ffn_blocks32 = static_cast<int32_t>(ffn_blocks);
  const auto max_row_blocks32 = static_cast<int32_t>(max_row_blocks);
  const int32_t num_blocks = static_cast<int32_t>((n + kChunk - 1) / kChunk);

  // All from torch's caching allocator: no cudaMalloc per call, and safe to
  // capture in a CUDA graph.
  auto block_counts = torch::empty({num_blocks, num_experts}, opts);
  auto tokens_per_expert = torch::empty({num_experts}, opts);
  auto padded_offsets = torch::empty({num_experts + 1}, opts);
  auto copy_row = torch::empty({n}, opts);
  auto row_token = torch::empty({max_rows}, opts);
  auto row_block_expert = torch::empty({max_row_blocks}, opts);
  auto row_offsets = torch::empty({max_row_blocks + 1}, opts);
  auto col_indices = torch::empty({max_row_blocks * ffn_blocks}, opts);
  auto error = torch::zeros({1}, opts);

  // Padding rows point at no token. 0xFF bytes = -1 as int32.
  C10_CUDA_CHECK(cudaMemsetAsync(row_token.data_ptr<int32_t>(), 0xFF, max_rows * sizeof(int32_t), stream));

  AT_DISPATCH_INDEX_TYPES(expert_idx.scalar_type(), "routing_count", [&] {
    count_kernel<index_t><<<num_blocks, kChunk, num_experts * sizeof(int32_t), stream>>>(
        expert_idx.data_ptr<index_t>(), n32, num_experts32, block_counts.data_ptr<int32_t>(),
        error.data_ptr<int32_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int offsets_threads = static_cast<int>((num_experts + 31) / 32 * 32);
  offsets_kernel<<<1, offsets_threads, 0, stream>>>(block_counts.data_ptr<int32_t>(), num_blocks, num_experts32,
                                                    block_size32, tokens_per_expert.data_ptr<int32_t>(),
                                                    padded_offsets.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_INDEX_TYPES(expert_idx.scalar_type(), "routing_scatter", [&] {
    scatter_kernel<index_t><<<num_blocks, kChunk, kWarps * num_experts * sizeof(int32_t), stream>>>(
        expert_idx.data_ptr<index_t>(), n32, num_experts32, static_cast<int32_t>(expert_idx.size(1)),
        block_counts.data_ptr<int32_t>(), padded_offsets.data_ptr<int32_t>(), copy_row.data_ptr<int32_t>(),
        row_token.data_ptr<int32_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int kLayoutThreads = 256;
  const int64_t layout_items = max_row_blocks * ffn_blocks + 1;  // +1: last row_offsets entry
  layout_kernel<<<static_cast<unsigned>((layout_items + kLayoutThreads - 1) / kLayoutThreads), kLayoutThreads, 0,
                  stream>>>(padded_offsets.data_ptr<int32_t>(), num_experts32, block_size32, ffn_blocks32,
                            max_row_blocks32, row_block_expert.data_ptr<int32_t>(), row_offsets.data_ptr<int32_t>(),
                            col_indices.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {tokens_per_expert, padded_offsets, row_token, copy_row, row_block_expert, row_offsets, col_indices, error};
}

}  // namespace sparsebench
