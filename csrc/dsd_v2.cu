// DSD v2: v1 + a cp.async pipeline, so loads overlap with math.
//
// ncu on v1: Tensor Cores busy 14%, top stall long_scoreboard (waiting on
// global memory). v1 loads a tile, waits, computes, repeats. Here, shared
// memory is a ring of STAGES tile buffers. While the warps compute tile i,
// the copies for tiles i+1 .. i+STAGES-1 are already in flight.
//
// Changed from v1: only how tiles get into shared memory. Fragment loads,
// mma and the epilogue are the same, bank conflicts included (v3 fixes those).
//
// STAGES = 3 for v2. With 2, just one load is in flight during a tile's
// math, which is too short to cover a trip to DRAM. bench_dsd.py also runs
// 2 stages (hidden version 21) to show that.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#include "dsd.hpp"
#include "mma.cuh"

namespace sparsebench {
namespace {

constexpr int kTileN = kDsdTileN;
constexpr int kTileK = 32;
constexpr int kThreads = 256;  // 8 warps
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kWarpN = kTileN / kWarpsN;
constexpr int kMmaN = kWarpN / 8;
constexpr int kStaticSmemLimit = 48 * 1024;

template <int BS, int STAGES>
__global__ void __launch_bounds__(kThreads) dsd_v2_kernel(const __nv_bfloat16* __restrict__ blocks,
                                                          const __nv_bfloat16* __restrict__ w,
                                                          const int32_t* __restrict__ row_offsets,
                                                          const int32_t* __restrict__ col_indices,
                                                          __nv_bfloat16* __restrict__ out, int32_t d_model) {
  constexpr int kWarpM = BS / kWarpsM;
  constexpr int kMmaM = kWarpM / 16;
  constexpr int kTilesPerBlock = BS / kTileK;
  static_assert(BS % kTileK == 0 && kWarpM % 16 == 0, "unsupported block size");
  static_assert(STAGES >= 2, "need at least 2 stages to overlap anything");
  static_assert(STAGES * (BS * kTileK + kTileK * kTileN) * sizeof(__nv_bfloat16) <= kStaticSmemLimit,
                "stages don't fit in static shared memory");

  __shared__ __align__(16) __nv_bfloat16 a_smem[STAGES][BS][kTileK];
  __shared__ __align__(16) __nv_bfloat16 b_smem[STAGES][kTileK][kTileN];

  const int row_block = blockIdx.x;
  const int n0 = blockIdx.y * kTileN;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int32_t begin = row_offsets[row_block];
  const int32_t end = row_offsets[row_block + 1];
  // One flat K loop over every 32-wide tile of every non-zero block in this row.
  const int32_t num_tiles = (end - begin) * kTilesPerBlock;

  // Start the copies for one K tile. Each thread issues its share; nothing waits.
  auto load_tile = [&](int32_t tile, int stage) {
    const int32_t p = begin + tile / kTilesPerBlock;
    const int k0 = (tile % kTilesPerBlock) * kTileK;
    const __nv_bfloat16* a_block = blocks + static_cast<int64_t>(p) * BS * BS;
    const __nv_bfloat16* w_rows = w + static_cast<int64_t>(col_indices[p]) * BS * d_model + n0;
    for (int i = tid; i < BS * kTileK / 8; i += kThreads) {
      const int r = i / (kTileK / 8);
      const int c = (i % (kTileK / 8)) * 8;
      cp_async_16(&a_smem[stage][r][c], a_block + static_cast<int64_t>(r) * BS + k0 + c);
    }
    for (int i = tid; i < kTileK * kTileN / 8; i += kThreads) {
      const int r = i / (kTileN / 8);
      const int c = (i % (kTileN / 8)) * 8;
      cp_async_16(&b_smem[stage][r][c], w_rows + static_cast<int64_t>(k0 + r) * d_model + c);
    }
  };

  float acc[kMmaM][kMmaN][4] = {};

  // Prologue: put the first STAGES-1 tiles in flight. Commit a group every
  // time, even an empty one (row shorter than the pipeline, or no tiles at
  // all). The wait below counts groups, so group i must always be tile i.
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    cp_async_commit();
  }

  for (int32_t i = 0; i < num_tiles; ++i) {
    // Groups committed so far: STAGES-1+i. Leaving STAGES-2 in flight means
    // groups 0..i are done, so tile i has landed (from this thread).
    cp_async_wait<STAGES - 2>();
    // One barrier does two jobs: tile i is now visible from every thread, and
    // every thread has finished computing tile i-1, so its stage
    // ((i-1) % STAGES, the one refilled next) is free to overwrite.
    __syncthreads();

    const int32_t next = i + STAGES - 1;
    if (next < num_tiles) load_tile(next, next % STAGES);
    cp_async_commit();

    const int stage = i % STAGES;
    // From here on, the same as v1's inner loop, on a_smem[stage] / b_smem[stage].
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      uint32_t a_frag[kMmaM][4];
      uint32_t b_frag[kMmaN][2];
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi) {
        const int r = warp_m * kWarpM + mi * 16 + g;
        a_frag[mi][0] = *reinterpret_cast<const uint32_t*>(&a_smem[stage][r][kk + 2 * t]);
        a_frag[mi][1] = *reinterpret_cast<const uint32_t*>(&a_smem[stage][r + 8][kk + 2 * t]);
        a_frag[mi][2] = *reinterpret_cast<const uint32_t*>(&a_smem[stage][r][kk + 2 * t + 8]);
        a_frag[mi][3] = *reinterpret_cast<const uint32_t*>(&a_smem[stage][r + 8][kk + 2 * t + 8]);
      }
#pragma unroll
      for (int ni = 0; ni < kMmaN; ++ni) {
        const int n = warp_n * kWarpN + ni * 8 + g;
        // k pair for one column is a row apart: two 16-bit loads (see v1).
        b_frag[ni][0] = pack_bf16(b_smem[stage][kk + 2 * t][n], b_smem[stage][kk + 2 * t + 1][n]);
        b_frag[ni][1] = pack_bf16(b_smem[stage][kk + 2 * t + 8][n], b_smem[stage][kk + 2 * t + 9][n]);
      }
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi)
#pragma unroll
        for (int ni = 0; ni < kMmaN; ++ni) mma_bf16_16816(acc[mi][ni], a_frag[mi], b_frag[ni]);
    }
  }
  // Only empty groups can be left, but don't exit with copies in flight.
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
  const dim3 grid(static_cast<unsigned>(row_offsets.numel() - 1), static_cast<unsigned>(d_model / kTileN));
  dsd_v2_kernel<BS, STAGES><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(blocks.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()), row_offsets.data_ptr<int32_t>(),
      col_indices.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), d_model);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int STAGES>
void launch_bs(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
               const torch::Tensor& col_indices, torch::Tensor& out) {
  switch (blocks.size(1)) {
    case 64: launch<64, STAGES>(blocks, w, row_offsets, col_indices, out); break;
    case 128: launch<128, STAGES>(blocks, w, row_offsets, col_indices, out); break;
    default: TORCH_CHECK(false, "dsd v2: unsupported block size ", blocks.size(1));
  }
}

}  // namespace

void dsd_v2_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out, int stages) {
  switch (stages) {
    case 2: launch_bs<2>(blocks, w, row_offsets, col_indices, out); break;
    case 3: launch_bs<3>(blocks, w, row_offsets, col_indices, out); break;
    default: TORCH_CHECK(false, "dsd v2: unsupported stage count ", stages);
  }
}

}  // namespace sparsebench
