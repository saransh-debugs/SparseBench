// DSD v5: v4 with a bigger inner loop, to close the gap to MegaBlocks.
//
// Phase 4: MegaBlocks' Triton DSD was 10-25% faster than v4, and grouped GEMM
// 1.6x faster on Mixtral-like shapes. ncu on v4: Tensor Cores busy ~54%,
// registers allow 2 thread blocks per SM, and each warp does little math per
// value it loads. The sparse layout and tile order were fine; the GEMM core
// was the gap.
//
// Changed from v4 (layout, row-block-first order, dead-row exit, cp.async
// ring and ldmatrix are the same):
//   thread-block tile  128 x 128, K step 32  ->  128 x 256, K step 64
//   warp tile          64 x 32               ->  64 x 64: 32 mma per 8 ldmatrix
//                                                per 16-wide k step (v4: 16 per 6)
//   shared memory      2 stages, 32 KB static ->  3 stages, 144 KB dynamic
//   blocks per SM      2                      ->  1, on purpose: the bigger
//                                                accumulator needs ~200 registers
//                                                per thread. Fewer, fatter warps
//                                                is the usual A100 GEMM trade.
// Half as many K steps also means half as many __syncthreads per unit of math.
//
// The inner loop lives in tile_gemm_v5.cuh, shared with SDD.
//
// Only block size 128 and D a multiple of 256. v4 showed 128 beats 64 on
// every shape, and real models' D is a multiple of 256.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "dsd.hpp"
#include "tile_gemm_v5.cuh"

namespace sparsebench {
namespace {

using namespace tile_v5;

constexpr int kBS = kTileM;  // block size = thread-block rows
constexpr int kTilesPerBlock = kBS / kTileK;

__global__ void __launch_bounds__(kThreads, 1) dsd_v5_kernel(const __nv_bfloat16* __restrict__ blocks,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             const int32_t* __restrict__ row_offsets,
                                                             const int32_t* __restrict__ col_indices,
                                                             __nv_bfloat16* __restrict__ out, int32_t d_model,
                                                             int32_t num_slices) {
  // Dynamic: 144 KB is past the 48 KB static limit. Raw bytes, because
  // extern __shared__ arrays of class types like __nv_bfloat16 aren't
  // reliably accepted by every nvcc. 128-byte aligned for the bank math.
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  const int row_block = static_cast<int>(blockIdx.x / num_slices);
  const int n0 = static_cast<int>(blockIdx.x % num_slices) * kTileN;
  const int tid = threadIdx.x;

  const int32_t begin = row_offsets[row_block];
  const int32_t end = row_offsets[row_block + 1];
  if (begin == end) return;  // dead row block: see v4

  // K tile t = 64 columns of non-zero block begin + t / 2, and the matching
  // 64 rows of W.
  auto load_tile = [&](int32_t tile, __nv_bfloat16* a_dst, __nv_bfloat16* b_dst) {
    const int32_t p = begin + tile / kTilesPerBlock;
    const int k0 = (tile % kTilesPerBlock) * kTileK;
    const __nv_bfloat16* a_block = blocks + static_cast<int64_t>(p) * kBS * kBS;
    const __nv_bfloat16* w_rows = w + static_cast<int64_t>(col_indices[p]) * kBS * d_model + n0;
#pragma unroll
    for (int j = 0; j < kAChunksPerThread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileK / kChunk);
      const int c = i % (kTileK / kChunk);
      cp_async_16(a_dst + a_offset(r, c), a_block + static_cast<int64_t>(r) * kBS + k0 + c * kChunk);
    }
#pragma unroll
    for (int j = 0; j < kBChunksPerThread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileN / kChunk);
      const int c = i % (kTileN / kChunk);
      cp_async_16(b_dst + b_offset(r, c), w_rows + static_cast<int64_t>(k0 + r) * d_model + c * kChunk);
    }
  };

  Acc acc;
  gemm_mainloop(smem, (end - begin) * kTilesPerBlock, load_tile, acc);

  const Lane pos;
#pragma unroll
  for (int mi = 0; mi < kMmaM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < kMmaN; ++ni) {
      const int c = n0 + pos.col(ni);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int64_t r = static_cast<int64_t>(row_block) * kBS + pos.row(mi, half);
        *reinterpret_cast<__nv_bfloat162*>(out + r * d_model + c) =
            __floats2bfloat162_rn(acc[mi][ni][2 * half], acc[mi][ni][2 * half + 1]);
      }
    }
  }
}

}  // namespace

void dsd_v5_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out) {
  TORCH_CHECK(blocks.size(1) == kBS && w.size(1) % kTileN == 0,
              "dsd v5 needs block size 128 and D a multiple of 256, got block size ", blocks.size(1), ", D ",
              w.size(1));

  opt_in_shared_memory(dsd_v5_kernel, kSmemBytes, "dsd v5");

  const auto d_model = static_cast<int32_t>(w.size(1));
  const int32_t num_slices = d_model / kTileN;
  const auto num_tiles = static_cast<unsigned>((row_offsets.numel() - 1) * num_slices);
  dsd_v5_kernel<<<num_tiles, kThreads, kSmemBytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(blocks.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()), row_offsets.data_ptr<int32_t>(),
      col_indices.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), d_model,
      num_slices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sparsebench
