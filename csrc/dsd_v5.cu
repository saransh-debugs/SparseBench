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
// Only block size 128 and D a multiple of 256. v4 showed 128 beats 64 on
// every shape, and real models' D is a multiple of 256.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "dsd.hpp"
#include "dsd_common.cuh"
#include "mma.cuh"

namespace sparsebench {
namespace {

using namespace tile_v5;

constexpr int kBS = 128;  // block size = thread-block rows
constexpr int kStages = 3;
constexpr int kWarpM = kBS / kWarpsM;  // 64 rows per warp
constexpr int kMmaM = kWarpM / 16;     // 4 mma tiles down
constexpr int kTilesPerBlock = kBS / kTileK;
constexpr int kStageElems = kBS * kTileK + kTileK * kTileN;  // A then B
constexpr int kSmemBytes = kStages * kStageElems * static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kSmemBytes == 144 * 1024, "3 stages x (16 KB A + 32 KB B)");
static_assert(kBS * kTileK / kChunk % kThreads == 0 && kTileK * kTileN / kChunk % kThreads == 0,
              "every thread copies the same number of chunks");

__global__ void __launch_bounds__(kThreads, 1) dsd_v5_kernel(const __nv_bfloat16* __restrict__ blocks,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             const int32_t* __restrict__ row_offsets,
                                                             const int32_t* __restrict__ col_indices,
                                                             __nv_bfloat16* __restrict__ out, int32_t d_model,
                                                             int32_t num_slices) {
  // Dynamic: 144 KB is past the 48 KB static limit. 128-byte aligned so the
  // bank math in dsd_common.cuh holds from the first row.
  // Declared as raw bytes: extern __shared__ arrays of class types like
  // __nv_bfloat16 aren't reliably accepted by every nvcc.
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  const int row_block = static_cast<int>(blockIdx.x / num_slices);
  const int n0 = static_cast<int>(blockIdx.x % num_slices) * kTileN;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int32_t begin = row_offsets[row_block];
  const int32_t end = row_offsets[row_block + 1];
  if (begin == end) return;  // dead row block: see v4
  const int32_t num_tiles = (end - begin) * kTilesPerBlock;

  auto a_stage = [&](int stage) { return smem + stage * kStageElems; };
  auto b_stage = [&](int stage) { return smem + stage * kStageElems + kBS * kTileK; };

  auto load_tile = [&](int32_t tile, int stage) {
    const int32_t p = begin + tile / kTilesPerBlock;
    const int k0 = (tile % kTilesPerBlock) * kTileK;
    const __nv_bfloat16* a_block = blocks + static_cast<int64_t>(p) * kBS * kBS;
    const __nv_bfloat16* w_rows = w + static_cast<int64_t>(col_indices[p]) * kBS * d_model + n0;
    __nv_bfloat16* a_dst = a_stage(stage);
    __nv_bfloat16* b_dst = b_stage(stage);
    // Fixed trip counts (4 and 8 chunks per thread) so the loops fully unroll.
#pragma unroll
    for (int j = 0; j < kBS * kTileK / kChunk / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileK / kChunk);
      const int c = i % (kTileK / kChunk);
      cp_async_16(a_dst + a_offset(r, c), a_block + static_cast<int64_t>(r) * kBS + k0 + c * kChunk);
    }
#pragma unroll
    for (int j = 0; j < kTileK * kTileN / kChunk / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (kTileN / kChunk);
      const int c = i % (kTileN / kChunk);
      cp_async_16(b_dst + b_offset(r, c), w_rows + static_cast<int64_t>(k0 + r) * d_model + c * kChunk);
    }
  };

  // ldmatrix lane -> row / chunk, as in v3 (see the comment there).
  const int a_lane_row = lane & 15;
  const int a_lane_chunk = lane >> 4;
  const int b_lane_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_lane_chunk = lane >> 4;

  float acc[kMmaM][kMmaN][4] = {};

  // Same pipeline as v2-v4 (comments in dsd_v2.cu).
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    cp_async_commit();
  }

  for (int32_t i = 0; i < num_tiles; ++i) {
    cp_async_wait<kStages - 2>();
    __syncthreads();

    const int32_t next = i + kStages - 1;
    if (next < num_tiles) load_tile(next, next % kStages);
    cp_async_commit();

    const __nv_bfloat16* a_s = a_stage(i % kStages);
    const __nv_bfloat16* b_s = b_stage(i % kStages);
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      uint32_t a_frag[kMmaM][4];
      uint32_t b_frag[kMmaN][2];
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi) {
        const int row = warp_m * kWarpM + mi * 16 + a_lane_row;
        ldmatrix_x4(a_frag[mi], a_s + a_offset(row, kk / kChunk + a_lane_chunk));
      }
#pragma unroll
      for (int nj = 0; nj < kMmaN / 2; ++nj) {
        const int chunk = (warp_n * kWarpN + nj * 16) / kChunk + b_lane_chunk;
        uint32_t r[4];
        ldmatrix_x4_trans(r, b_s + b_offset(kk + b_lane_row, chunk));
        b_frag[2 * nj][0] = r[0];
        b_frag[2 * nj][1] = r[1];
        b_frag[2 * nj + 1][0] = r[2];
        b_frag[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi)
#pragma unroll
        for (int ni = 0; ni < kMmaN; ++ni) mma_bf16_16816(acc[mi][ni], a_frag[mi], b_frag[ni]);
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int mi = 0; mi < kMmaM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < kMmaN; ++ni) {
      const int64_t r = static_cast<int64_t>(row_block) * kBS + warp_m * kWarpM + mi * 16 + g;
      const int c = n0 + warp_n * kWarpN + ni * 8 + 2 * t;
      *reinterpret_cast<__nv_bfloat162*>(out + r * d_model + c) =
          __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<__nv_bfloat162*>(out + (r + 8) * d_model + c) =
          __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
    }
  }
}

}  // namespace

void dsd_v5_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out) {
  TORCH_CHECK(blocks.size(1) == kBS && w.size(1) % kTileN == 0,
              "dsd v5 needs block size 128 and D a multiple of 256, got block size ", blocks.size(1), ", D ",
              w.size(1));

  // Past 48 KB, a kernel must opt in to dynamic shared memory, and the GPU
  // must have that much: A100 / H100 do, T4 (64 KB) doesn't.
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(max_optin >= kSmemBytes, "dsd v5 needs ", kSmemBytes, " bytes of shared memory per block; this GPU allows ",
              max_optin);
  // Set on every launch, not once: function attributes can be per-device, and
  // the call costs well under a microsecond next to a millisecond kernel.
  C10_CUDA_CHECK(cudaFuncSetAttribute(dsd_v5_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

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
