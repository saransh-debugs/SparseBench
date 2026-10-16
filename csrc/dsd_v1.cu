// DSD v1: out[P, D] = H[P, E*F] (block-sparse, BSR) @ W[E*F, D] (dense).
//
// The plain baseline. Correct, uses Tensor Cores, and makes three choices on
// purpose that later versions fix, so each fix shows up in ncu:
//   - Load, sync, compute, sync. Loads and math never overlap.   (v2: cp.async)
//   - B fragments come from 16-bit shared loads, with bank
//     conflicts.                                           (v3: swizzle + ldmatrix)
//   - Every row block gets a thread block, even past the real
//     end, which only writes zeros.                  (v4: sparse-aware scheduling)
//
// One thread block computes a BS x 128 tile of out: one row block (so one
// expert) by one 128-column slice of D. It walks that row's non-zero blocks
// and, inside each, K in steps of 32.

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
constexpr int kWarpN = kTileN / kWarpsN;  // 32 columns per warp
constexpr int kMmaN = kWarpN / 8;         // 4 mma tiles across

template <int BS>
__global__ void __launch_bounds__(kThreads) dsd_v1_kernel(const __nv_bfloat16* __restrict__ blocks,
                                                          const __nv_bfloat16* __restrict__ w,
                                                          const int32_t* __restrict__ row_offsets,
                                                          const int32_t* __restrict__ col_indices,
                                                          __nv_bfloat16* __restrict__ out, int32_t d_model) {
  constexpr int kWarpM = BS / kWarpsM;  // 64 rows per warp for BS=128, 32 for BS=64
  constexpr int kMmaM = kWarpM / 16;
  static_assert(BS % kTileK == 0 && kWarpM % 16 == 0, "unsupported block size");

  __shared__ __align__(16) __nv_bfloat16 a_smem[BS][kTileK];      // H tile, row-major (k contiguous)
  __shared__ __align__(16) __nv_bfloat16 b_smem[kTileK][kTileN];  // W tile, row-major (n contiguous)

  const int row_block = blockIdx.x;
  const int n0 = blockIdx.y * kTileN;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const int g = lane >> 2;  // mma "group": row (A, D) or column (B)
  const int t = lane & 3;   // index inside the group: picks the k pair

  float acc[kMmaM][kMmaN][4] = {};

  const int32_t begin = row_offsets[row_block];
  const int32_t end = row_offsets[row_block + 1];
  for (int32_t p = begin; p < end; ++p) {
    // Block p of H, and the BS rows of W it multiplies.
    const __nv_bfloat16* a_block = blocks + static_cast<int64_t>(p) * BS * BS;
    const __nv_bfloat16* w_rows = w + static_cast<int64_t>(col_indices[p]) * BS * d_model + n0;

    for (int k0 = 0; k0 < BS; k0 += kTileK) {
      // Global -> registers -> shared, 16 bytes (8 bf16) per access.
      for (int i = tid; i < BS * kTileK / 8; i += kThreads) {
        const int r = i / (kTileK / 8);
        const int c = (i % (kTileK / 8)) * 8;
        *reinterpret_cast<uint4*>(&a_smem[r][c]) =
            *reinterpret_cast<const uint4*>(a_block + static_cast<int64_t>(r) * BS + k0 + c);
      }
      for (int i = tid; i < kTileK * kTileN / 8; i += kThreads) {
        const int r = i / (kTileN / 8);
        const int c = (i % (kTileN / 8)) * 8;
        *reinterpret_cast<uint4*>(&b_smem[r][c]) =
            *reinterpret_cast<const uint4*>(w_rows + static_cast<int64_t>(k0 + r) * d_model + c);
      }
      __syncthreads();

      // Fully unrolled so acc / a_frag / b_frag stay in registers. A loop
      // index the compiler can't resolve would push them to local memory.
#pragma unroll
      for (int kk = 0; kk < kTileK; kk += 16) {
        uint32_t a_frag[kMmaM][4];
        uint32_t b_frag[kMmaN][2];
#pragma unroll
        for (int mi = 0; mi < kMmaM; ++mi) {
          const int r = warp_m * kWarpM + mi * 16 + g;
          // H is row-major, so each k pair is adjacent: one 32-bit load.
          a_frag[mi][0] = *reinterpret_cast<const uint32_t*>(&a_smem[r][kk + 2 * t]);
          a_frag[mi][1] = *reinterpret_cast<const uint32_t*>(&a_smem[r + 8][kk + 2 * t]);
          a_frag[mi][2] = *reinterpret_cast<const uint32_t*>(&a_smem[r][kk + 2 * t + 8]);
          a_frag[mi][3] = *reinterpret_cast<const uint32_t*>(&a_smem[r + 8][kk + 2 * t + 8]);
        }
#pragma unroll
        for (int ni = 0; ni < kMmaN; ++ni) {
          const int n = warp_n * kWarpN + ni * 8 + g;
          // W is row-major, so the k pair for one column is a whole row
          // apart. A 32-bit load here would grab two columns instead: wrong
          // answer, no error. Two 16-bit loads, packed.
          b_frag[ni][0] = pack_bf16(b_smem[kk + 2 * t][n], b_smem[kk + 2 * t + 1][n]);
          b_frag[ni][1] = pack_bf16(b_smem[kk + 2 * t + 8][n], b_smem[kk + 2 * t + 9][n]);
        }
#pragma unroll
        for (int mi = 0; mi < kMmaM; ++mi)
#pragma unroll
          for (int ni = 0; ni < kMmaN; ++ni) mma_bf16_16816(acc[mi][ni], a_frag[mi], b_frag[ni]);
      }
      __syncthreads();  // everyone done reading before the next load overwrites
    }
  }

  // fp32 -> bf16, two adjacent columns per store. A row block with no
  // non-zero blocks never entered the loop, so it writes zeros.
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

template <int BS>
void launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
            const torch::Tensor& col_indices, torch::Tensor& out) {
  const auto d_model = static_cast<int32_t>(w.size(1));
  const dim3 grid(static_cast<unsigned>(row_offsets.numel() - 1), static_cast<unsigned>(d_model / kTileN));
  dsd_v1_kernel<BS><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(blocks.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()), row_offsets.data_ptr<int32_t>(),
      col_indices.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), d_model);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void dsd_v1_launch(const torch::Tensor& blocks, const torch::Tensor& w, const torch::Tensor& row_offsets,
                   const torch::Tensor& col_indices, torch::Tensor& out) {
  switch (blocks.size(1)) {
    case 64: launch<64>(blocks, w, row_offsets, col_indices, out); break;
    case 128: launch<128>(blocks, w, row_offsets, col_indices, out); break;
    default: TORCH_CHECK(false, "dsd v1: unsupported block size ", blocks.size(1));
  }
}

}  // namespace sparsebench
