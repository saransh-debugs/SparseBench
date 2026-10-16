#pragma once
// The v5 inner loop, shared by DSD v5 and SDD.
//
// One thread block computes a 128 x 256 fp32 tile:  acc += A[128, K] @ B[K, 256].
// The caller says how to fetch K tile `t` of A and B (that's the only thing
// DSD and SDD do differently); this file owns everything else:
//   - a 3-stage cp.async ring in dynamic shared memory (pipeline: dsd_v2.cu)
//   - the XOR swizzle and ldmatrix fragment loads (bank math: dsd_common.cuh)
//   - 64 x 64 warp tiles of mma.sync m16n8k16 (why these sizes: dsd_v5.cu)
//
// Shared-memory layout per stage: A tile [128][64] (swizzled), then B tile
// [64][256] (swizzled). The caller's load function writes chunks with
// a_offset / b_offset; this file reads them back the same way.

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "dsd_common.cuh"
#include "mma.cuh"

namespace sparsebench::tile_v5 {

constexpr int kTileM = 128;
constexpr int kStages = 3;
constexpr int kWarpM = kTileM / kWarpsM;  // 64 rows per warp
constexpr int kMmaM = kWarpM / 16;        // 4 mma tiles down
constexpr int kStageElems = kTileM * kTileK + kTileK * kTileN;
constexpr int kSmemBytes = kStages * kStageElems * static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kSmemBytes == 144 * 1024, "3 stages x (16 KB A + 32 KB B)");

// Chunks of 16 bytes each thread copies per K tile. Fixed counts, so the
// callers' copy loops fully unroll.
constexpr int kAChunksPerThread = kTileM * kTileK / kChunk / kThreads;  // 4
constexpr int kBChunksPerThread = kTileK * kTileN / kChunk / kThreads;  // 8
static_assert(kTileM * kTileK / kChunk % kThreads == 0 && kTileK * kTileN / kChunk % kThreads == 0,
              "every thread copies the same number of chunks");

using Acc = float[kMmaM][kMmaN][4];

// Where this thread's accumulator values sit in the 128 x 256 tile:
// acc[mi][ni][0..1] -> row  warp_m*64 + mi*16 + g,      cols c, c+1
// acc[mi][ni][2..3] -> row  warp_m*64 + mi*16 + g + 8,  cols c, c+1
// with c = warp_n*64 + ni*8 + 2t (the mma D-fragment layout, see mma.cuh).
struct Lane {
  int warp_m, warp_n, g, t;
  __device__ __forceinline__ Lane() {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    warp_m = warp / kWarpsN;
    warp_n = warp % kWarpsN;
    g = lane >> 2;
    t = lane & 3;
  }
  __device__ __forceinline__ int row(int mi, int half) const { return warp_m * kWarpM + mi * 16 + g + 8 * half; }
  __device__ __forceinline__ int col(int ni) const { return warp_n * kWarpN + ni * 8 + 2 * t; }
};

// load_tile(tile, a_dst, b_dst): start cp.async copies of K tile `tile` into
// the given stage buffers (each thread its own chunks). Must not wait.
template <typename LoadTile>
__device__ __forceinline__ void gemm_mainloop(__nv_bfloat16* smem, int32_t num_tiles, LoadTile&& load_tile,
                                              Acc& acc) {
  const int lane = threadIdx.x & 31;
  const Lane pos;
  auto a_stage = [&](int stage) { return smem + stage * kStageElems; };
  auto b_stage = [&](int stage) { return smem + stage * kStageElems + kTileM * kTileK; };

  // ldmatrix lane -> row / chunk (see dsd_v3.cu).
  const int a_lane_row = lane & 15;
  const int a_lane_chunk = lane >> 4;
  const int b_lane_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_lane_chunk = lane >> 4;

#pragma unroll
  for (int mi = 0; mi < kMmaM; ++mi)
#pragma unroll
    for (int ni = 0; ni < kMmaN; ++ni)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[mi][ni][v] = 0.0f;

  // Prologue and main loop: the v2 pipeline (comments in dsd_v2.cu). Empty
  // commits keep group i == tile i when there are fewer tiles than stages.
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < num_tiles) load_tile(s, a_stage(s), b_stage(s));
    cp_async_commit();
  }

  for (int32_t i = 0; i < num_tiles; ++i) {
    cp_async_wait<kStages - 2>();
    __syncthreads();

    const int32_t next = i + kStages - 1;
    if (next < num_tiles) load_tile(next, a_stage(next % kStages), b_stage(next % kStages));
    cp_async_commit();

    const __nv_bfloat16* a_s = a_stage(i % kStages);
    const __nv_bfloat16* b_s = b_stage(i % kStages);
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      uint32_t a_frag[kMmaM][4];
      uint32_t b_frag[kMmaN][2];
#pragma unroll
      for (int mi = 0; mi < kMmaM; ++mi) {
        const int row = pos.warp_m * kWarpM + mi * 16 + a_lane_row;
        ldmatrix_x4(a_frag[mi], a_s + a_offset(row, kk / kChunk + a_lane_chunk));
      }
#pragma unroll
      for (int nj = 0; nj < kMmaN / 2; ++nj) {
        const int chunk = (pos.warp_n * kWarpN + nj * 16) / kChunk + b_lane_chunk;
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
}

// Host side. Past 48 KB a kernel must opt in to dynamic shared memory, and the
// GPU must have that much: A100 / H100 do, T4 (64 KB) doesn't. Called on every
// launch, not once: function attributes can be per-device, and the call costs
// well under a microsecond next to a millisecond kernel.
template <typename Kernel>
inline void opt_in_shared_memory(Kernel kernel, int bytes, const char* name) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(max_optin >= bytes, name, " needs ", bytes, " bytes of shared memory per block; this GPU allows ",
              max_optin);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace sparsebench::tile_v5
