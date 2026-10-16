#pragma once
// Tensor Core helpers shared by every DSD kernel version.
//
// mma.sync.m16n8k16 with bf16 inputs and fp32 accumulators. Per warp:
//   D[16x8] += A[16x16] @ B[16x8]
// Each lane holds pieces of A, B and D in registers. With g = lane / 4 and
// t = lane % 4 (PTX ISA, "Matrix Fragments for mma.m16n8k16"):
//   A, 4 regs of 2 bf16:  a0 (row g,   k 2t..2t+1)    a1 (row g+8, k 2t..2t+1)
//                         a2 (row g,   k 2t+8..2t+9)  a3 (row g+8, k 2t+8..2t+9)
//   B, 2 regs of 2 bf16:  b0 (k 2t..2t+1,   col g)    b1 (k 2t+8..2t+9, col g)
//   D, 4 fp32:            d0,d1 (row g, col 2t..2t+1) d2,d3 (row g+8, col 2t..2t+1)
// In every 2-element register, the lower k (or col) sits in the low 16 bits.
//
// Raw PTX instead of the WMMA API: WMMA hides which lane holds what, and the
// later versions load fragments with ldmatrix, which needs exactly this layout.

#include <cuda_bf16.h>

#include <cstdint>

namespace sparsebench {

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// lo goes in bits 0-15, hi in bits 16-31.
__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

// ---- cp.async (sm_80+): global -> shared without going through registers ----
//
// A plain load makes the thread wait (long_scoreboard stall) the moment it
// needs the value, and costs a register plus a store instruction. cp.async
// hands the copy to the memory system and returns at once. The thread keeps
// computing and only waits when it calls cp_async_wait.
//
// Copies are tracked in "groups": every cp_async_commit closes one group.
// cp_async_wait<N> blocks until at most N of this thread's groups are still
// in flight. It covers only this thread's copies, so a __syncthreads must
// follow before reading what other threads copied.

// .cg = cache in L2 only. Each tile is read once per thread block, so
// putting it in L1 too would just evict data that does get reused.
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const auto smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

// The "memory" clobber stops the compiler from moving shared-memory reads
// above the wait.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

}  // namespace sparsebench
