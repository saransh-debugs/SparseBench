// DSD v4: v3 + sparse-aware tile order + skip dead row blocks.
//
// ncu on v3: bank conflicts gone, but GPU memory bandwidth 78% used. The grid
// ran column slice by column slice, so every row block of H was re-read from
// GPU memory once per slice (8x fine-grained, 32x Mixtral): by the time the
// next slice came back to it, it had left the 40 MB L2.
//
// Changed from v3 (fragment loads, swizzle, mma, epilogue are the same):
//   1. 1-D grid, row block first: tile id -> (id / slices, id % slices). All
//      column slices of a row block run together, so H's row block comes from
//      GPU memory once and the other slices hit L2. Row blocks are sorted by
//      expert, so neighbours share W slices and reuse them from L2 too.
//   2. A row block with no non-zero blocks (worst-case padding past the real
//      end) exits at once and writes nothing. Its rows are left unwritten.
//   3. 2 stages instead of 3: v3 measured 2 as faster (up to 10%).
//
// No persistent kernel / work queue: every real row block has the same work
// (F / bs blocks), so there is no uneven work for a queue to balance. Skewed
// routing costs padding (more row blocks), not imbalance.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "dsd.hpp"
#include "dsd_common.cuh"
#include "mma.cuh"

namespace sparsebench {
namespace {

using namespace tile;  // tile sizes and the a_offset / b_offset swizzle

template <int BS, int STAGES>
__global__ void __launch_bounds__(kThreads) dsd_v4_kernel(const __nv_bfloat16* __restrict__ blocks,
                                                          const __nv_bfloat16* __restrict__ w,
                                                          const int32_t* __restrict__ row_offsets,
                                                          const int32_t* __restrict__ col_indices,
                                                          __nv_bfloat16* __restrict__ out, int32_t d_model,
                                                          int32_t num_slices) {
  constexpr int kWarpM = BS / kWarpsM;
  constexpr int kMmaM = kWarpM / 16;
  constexpr int kTilesPerBlock = BS / kTileK;
  static_assert(BS % kTileK == 0 && kWarpM % 16 == 0, "unsupported block size");
  static_assert(kMmaN % 2 == 0, "B fragments are loaded two mma tiles at a time");
  static_assert(STAGES >= 2, "need at least 2 stages to overlap anything");
  static_assert(STAGES * (BS * kTileK + kTileK * kTileN) * sizeof(__nv_bfloat16) <= kStaticSmemLimit,
                "stages don't fit in static shared memory");

  // 128-byte aligned so "slot = address / 16 % 8" holds from the first row.
  __shared__ __align__(128) __nv_bfloat16 a_smem[STAGES][BS * kTileK];
  __shared__ __align__(128) __nv_bfloat16 b_smem[STAGES][kTileK * kTileN];

  // Row block first: all column slices of one row block are neighbours in
  // launch order, so they run at the same time and share H through L2.
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
  // Dead row block: nothing to compute, and its rows are never read. The
  // whole thread block leaves together, before any barrier.
  if (begin == end) return;
  const int32_t num_tiles = (end - begin) * kTilesPerBlock;

  // Same copies as v2; each chunk goes to its swizzled slot.
  auto load_tile = [&](int32_t tile, int stage) {
    const int32_t p = begin + tile / kTilesPerBlock;
    const int k0 = (tile % kTilesPerBlock) * kTileK;
    const __nv_bfloat16* a_block = blocks + static_cast<int64_t>(p) * BS * BS;
    const __nv_bfloat16* w_rows = w + static_cast<int64_t>(col_indices[p]) * BS * d_model + n0;
    for (int i = tid; i < BS * kTileK / kChunk; i += kThreads) {
      const int r = i / (kTileK / kChunk);
      const int c = i % (kTileK / kChunk);
      cp_async_16(&a_smem[stage][a_offset(r, c)], a_block + static_cast<int64_t>(r) * BS + k0 + c * kChunk);
    }
    for (int i = tid; i < kTileK * kTileN / kChunk; i += kThreads) {
      const int r = i / (kTileN / kChunk);
      const int c = i % (kTileN / kChunk);
      cp_async_16(&b_smem[stage][b_offset(r, c)], w_rows + static_cast<int64_t>(k0 + r) * d_model + c * kChunk);
    }
  };

  // Which row each lane points ldmatrix at. Fixed for the whole kernel.
  //   A (x4): tiles = rows 0-7 / 8-15 of the 16-row mma tile, at k 0-7 / 8-15.
  //           Lane L: row L % 16, k chunk L / 16.
  //   B (x4.trans): tiles = k 0-7 / 8-15, for n 0-7 / 8-15 (two mma tiles).
  //           Lane L: k row (L % 8) + 8 * ((L / 8) % 2), n chunk L / 16.
  const int a_lane_row = lane & 15;
  const int a_lane_chunk = lane >> 4;
  const int b_lane_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_lane_chunk = lane >> 4;

  float acc[kMmaM][kMmaN][4] = {};

  // Prologue and main loop: identical to v2/v3 (see dsd_v2.cu for the comments).
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    cp_async_commit();
  }

  for (int32_t i = 0; i < num_tiles; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    const int32_t next = i + STAGES - 1;
    if (next < num_tiles) load_tile(next, next % STAGES);
    cp_async_commit();

    const __nv_bfloat16* a_stage = a_smem[i % STAGES];
    const __nv_bfloat16* b_stage = b_smem[i % STAGES];
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      uint32_t a_frag[kMmaM][4];
      uint32_t b_frag[kMmaN][2];
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi) {
        const int row = warp_m * kWarpM + mi * 16 + a_lane_row;
        ldmatrix_x4(a_frag[mi], a_stage + a_offset(row, kk / kChunk + a_lane_chunk));
      }
#pragma unroll
      for (int nj = 0; nj < kMmaN / 2; ++nj) {
        const int chunk = (warp_n * kWarpN + nj * 16) / kChunk + b_lane_chunk;
        uint32_t r[4];
        ldmatrix_x4_trans(r, b_stage + b_offset(kk + b_lane_row, chunk));
        b_frag[2 * nj][0] = r[0];      // k 0-7,  n 0-7
        b_frag[2 * nj][1] = r[1];      // k 8-15, n 0-7
        b_frag[2 * nj + 1][0] = r[2];  // k 0-7,  n 8-15
        b_frag[2 * nj + 1][1] = r[3];  // k 8-15, n 8-15
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
      const int64_t r = static_cast<int64_t>(row_block) * BS + warp_m * kWarpM + mi * 16 + g;
      const int c = n0 + warp_n * kWarpN + ni * 8 + 2 * t;
      *reinterpret_cast<__nv_bfloat162*>(out + r * d_model + c) =
          __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<__nv_bfloat162*>(out + (r + 8) * d_model + c) =
          __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
    }
  }
}

template <int BS, int STAGES>
void launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
            const torch::Tensor& col_indices, torch::Tensor& out) {
  const auto d_model = static_cast<int32_t>(w.size(1));
  const int32_t num_slices = d_model / kTileN;
  // dsd.cpp checked that row blocks * slices fits in grid x.
  const auto num_tiles = static_cast<unsigned>((row_offsets.numel() - 1) * num_slices);
  dsd_v4_kernel<BS, STAGES><<<num_tiles, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(blocks.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()), row_offsets.data_ptr<int32_t>(),
      col_indices.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), d_model,
      num_slices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int STAGES>
void launch_bs(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
               const torch::Tensor& col_indices, torch::Tensor& out) {
  switch (blocks.size(1)) {
    case 64: launch<64, STAGES>(blocks, w, row_offsets, col_indices, out); break;
    case 128: launch<128, STAGES>(blocks, w, row_offsets, col_indices, out); break;
    default: TORCH_CHECK(false, "dsd v4: unsupported block size ", blocks.size(1));
  }
}

}  // namespace

void dsd_v4_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out, int stages) {
  switch (stages) {
    case 2: launch_bs<2>(blocks, w, row_offsets, col_indices, out); break;
    case 3: launch_bs<3>(blocks, w, row_offsets, col_indices, out); break;
    default: TORCH_CHECK(false, "dsd v4: unsupported stage count ", stages);
  }
}

}  // namespace sparsebench
