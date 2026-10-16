// SDD: the MoE's first matmul, written straight into the block-sparse layout.
//
//   H blocks [nnz, 128, 128] = gelu( X_grouped [P, D] @ W1[expert] [D, F] ),
//   computed only on the layout's blocks.
//
// Row block r belongs to one expert e and owns blocks row_offsets[r] + j,
// j = 0 .. F/128 - 1, holding columns j*128 .. j*128+127 of e's FFN (the
// layout build_topology / routing.cu produce). So each row block is a dense
// GEMM: M = 128 padded rows, N = F, K = D. The inner loop is v5's
// (tile_gemm_v5.cuh); only the edges are SDD-specific:
//
//   A rows   Gathered on the fly: padded row -> token via row_token, and
//            cp.async reads x[token] directly. No grouped copy of X is ever
//            made (the reference builds one: group_tokens). Padding rows
//            (row_token = -1) use cp.async's zero-fill form: no memory reads.
//   B        W1 [E, D, F] as stored, rows of the row block's expert. No
//            [D, E*F] concatenated copy (the reference makes one).
//   Output   GELU fused into the epilogue (exact erf form, like F.gelu), then
//            each 128 x 256 tile lands as two 128 x 128 blocks in BSR order,
//            ready for DSD. No separate pass over H for the activation.
//   Grid     Row block first, dead row blocks exit (as in DSD v4/v5).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include "ops.hpp"
#include "tile_gemm_v5.cuh"

namespace sparsebench {
namespace {

using namespace tile_v5;

constexpr int kBS = kTileM;
constexpr int kSddSmemBytes = tile_v5::kSmemBytes + kBS * static_cast<int>(sizeof(int32_t));  // + token table

__device__ __forceinline__ float gelu_erf(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }

__global__ void __launch_bounds__(kThreads, 1) sdd_kernel(const __nv_bfloat16* __restrict__ x,
                                                          const __nv_bfloat16* __restrict__ w1,
                                                          const int32_t* __restrict__ row_token,
                                                          const int32_t* __restrict__ row_block_expert,
                                                          const int32_t* __restrict__ row_offsets,
                                                          __nv_bfloat16* __restrict__ out, int32_t d_model,
                                                          int32_t ffn_dim, int32_t num_slices, bool apply_gelu) {
  extern __shared__ __align__(128) unsigned char smem_raw[];  // raw bytes: see dsd_v5.cu
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  int32_t* tokens = reinterpret_cast<int32_t*>(smem_raw + tile_v5::kSmemBytes);  // [128] after the stages

  const int row_block = static_cast<int>(blockIdx.x / num_slices);
  const int n0 = static_cast<int>(blockIdx.x % num_slices) * kTileN;
  const int tid = threadIdx.x;

  const int32_t begin = row_offsets[row_block];
  if (begin == row_offsets[row_block + 1]) return;  // dead row block: no blocks to write
  const int32_t expert = row_block_expert[row_block];

  // Each row's token, looked up once per thread block instead of once per
  // copy per K tile. The barrier inside gemm_mainloop's first iteration
  // comes after the prologue's copies, so publish the table first.
  if (tid < kBS) tokens[tid] = row_token[static_cast<int64_t>(row_block) * kBS + tid];
  __syncthreads();

  const __nv_bfloat16* w_cols = w1 + static_cast<int64_t>(expert) * d_model * ffn_dim + n0;
  auto load_tile = [&](int32_t tile, __nv_bfloat16* a_dst, __nv_bfloat16* b_dst) {
    const int k0 = tile * kTileK;
#pragma unroll
    for (int j = 0; j < kAChunksPerThread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileK / kChunk);
      const int c = i % (kTileK / kChunk);
      const int32_t token = tokens[r];
      // Padding rows still pass a valid address (x itself); zero-fill reads nothing.
      const __nv_bfloat16* src = x + (token >= 0 ? static_cast<int64_t>(token) * d_model + k0 + c * kChunk : 0);
      cp_async_16_zfill(a_dst + a_offset(r, c), src, token >= 0);
    }
#pragma unroll
    for (int j = 0; j < kBChunksPerThread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileN / kChunk);
      const int c = i % (kTileN / kChunk);
      cp_async_16(b_dst + b_offset(r, c), w_cols + static_cast<int64_t>(k0 + r) * ffn_dim + c * kChunk);
    }
  };

  Acc acc;
  gemm_mainloop(smem, d_model / kTileK, load_tile, acc);

  // Column f of the expert's FFN -> block begin + f / 128, column f % 128.
  // Two adjacent columns (2t, 2t+1) never straddle a block edge: 128 is even.
  const Lane pos;
#pragma unroll
  for (int mi = 0; mi < kMmaM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < kMmaN; ++ni) {
      const int f = n0 + pos.col(ni);
      __nv_bfloat16* block = out + (static_cast<int64_t>(begin) + f / kBS) * kBS * kBS;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float v0 = acc[mi][ni][2 * half];
        float v1 = acc[mi][ni][2 * half + 1];
        if (apply_gelu) {
          v0 = gelu_erf(v0);
          v1 = gelu_erf(v1);
        }
        *reinterpret_cast<__nv_bfloat162*>(block + pos.row(mi, half) * kBS + f % kBS) = __floats2bfloat162_rn(v0, v1);
      }
    }
  }
}

void check_index(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 1 && t.is_contiguous(), name,
              " must be a contiguous 1-D int32 CUDA tensor");
}

}  // namespace

torch::Tensor sdd_cuda(const torch::Tensor& x, const torch::Tensor& w1, const torch::Tensor& row_token,
                       const torch::Tensor& row_block_expert, const torch::Tensor& row_offsets, int64_t num_nnz,
                       bool apply_gelu) {
  TORCH_CHECK(x.is_cuda() && w1.is_cuda() && x.device() == w1.device(), "x and w1 must be on the same GPU");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w1.scalar_type() == torch::kBFloat16,
              "x and w1 must be bfloat16");
  TORCH_CHECK(x.is_contiguous() && w1.is_contiguous(), "x and w1 must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w1.dim() == 3 && w1.size(1) == x.size(1), "x must be [T, D] and w1 [E, D, F], got ",
              x.sizes(), " and ", w1.sizes());
  check_index(row_token, "row_token");
  check_index(row_block_expert, "row_block_expert");
  check_index(row_offsets, "row_offsets");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w1.data_ptr()) % 16 == 0,
              "x and w1 must be 16-byte aligned");

  const int64_t d_model = x.size(1);
  const int64_t ffn_dim = w1.size(2);
  const int64_t num_row_blocks = row_offsets.numel() - 1;
  TORCH_CHECK(d_model > 0 && d_model % kTileK == 0, "sdd needs D (", d_model, ") to be a multiple of ", kTileK);
  TORCH_CHECK(ffn_dim > 0 && ffn_dim % kTileN == 0, "sdd needs F (", ffn_dim, ") to be a multiple of ", kTileN);
  TORCH_CHECK(row_token.numel() == num_row_blocks * kBS, "sdd needs block size 128: row_token has ",
              row_token.numel(), " rows for ", num_row_blocks, " row blocks");
  TORCH_CHECK(row_block_expert.numel() == num_row_blocks, "row_block_expert must have one entry per row block");
  // The layout must be one block per (row block, 128 FFN columns), as
  // build_topology / routing.cu make it. Checked by size; the values can't be
  // checked without a GPU sync.
  TORCH_CHECK(num_nnz == num_row_blocks * (ffn_dim / kBS), "expected ", num_row_blocks * (ffn_dim / kBS),
              " blocks (row blocks x F/128), got ", num_nnz);
  TORCH_CHECK(num_row_blocks * (ffn_dim / kTileN) <= std::numeric_limits<int32_t>::max() &&
                  ffn_dim * d_model <= std::numeric_limits<int32_t>::max(),
              "problem too large");

  auto out = torch::empty({num_nnz, kBS, kBS}, w1.options());
  if (num_row_blocks == 0) return out;

  const c10::cuda::CUDAGuard guard(x.device());
  opt_in_shared_memory(sdd_kernel, kSddSmemBytes, "sdd");
  const auto num_slices = static_cast<int32_t>(ffn_dim / kTileN);
  const auto num_tiles = static_cast<unsigned>(num_row_blocks * num_slices);
  sdd_kernel<<<num_tiles, kThreads, kSddSmemBytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr<at::BFloat16>()), row_token.data_ptr<int32_t>(),
      row_block_expert.data_ptr<int32_t>(), row_offsets.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int32_t>(d_model),
      static_cast<int32_t>(ffn_dim), num_slices, apply_gelu);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace sparsebench
