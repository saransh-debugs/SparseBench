#pragma once
// Tile shape and shared-memory swizzle shared by DSD v3 and later.
// (v1 and v2 keep their own copies: they are frozen as measured.)
//
// Bank model: shared memory serves 128 bytes per transaction, as 8 slots of
// 16 bytes. An ldmatrix phase asks for 8 rows of 16 bytes. If two of them
// fall in the same slot (different addresses), they're served one after another.
//
// A tile [BS][32] bf16: a row is 64 bytes = 4 chunks, slot = (4*row + chunk) % 8.
//   8 consecutive rows, same chunk -> slots c, c+4, c, c+4, ...: 4-way conflict.
//   Stored chunk = chunk ^ ((row >> 1) & 3) -> slot = 4*(row & 1) + (chunk ^ (row>>1 & 3)),
//   which takes 8 different values over 8 consecutive rows. 0 conflicts.
// B tile [32][128] bf16: a row is 256 bytes = 16 chunks, slot = chunk % 8.
//   8 consecutive k-rows, same chunk -> same slot 8 times: 8-way conflict.
//   Stored chunk = chunk ^ (row & 7) -> slot = (chunk ^ row) % 8: 8 different slots.
//
// Why XOR and not padding each row: padding costs shared memory, and 3 stages
// already fill the 48 KB static limit. XOR costs no memory, one instruction.

#include "dsd.hpp"

namespace sparsebench::tile {

constexpr int kTileN = kDsdTileN;
constexpr int kTileK = 32;
constexpr int kThreads = 256;  // 8 warps
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kWarpN = kTileN / kWarpsN;
constexpr int kMmaN = kWarpN / 8;
constexpr int kStaticSmemLimit = 48 * 1024;
constexpr int kChunk = 8;  // bf16 per 16-byte chunk

// Element offset of 16-byte chunk `chunk` of row `row`, after the swizzle.
__device__ __forceinline__ int a_offset(int row, int chunk) {
  return row * kTileK + ((chunk ^ ((row >> 1) & 3)) * kChunk);
}
__device__ __forceinline__ int b_offset(int row, int chunk) {
  return row * kTileN + ((chunk ^ (row & 7)) * kChunk);
}

}  // namespace sparsebench::tile

// ---- v5: bigger tiles (128 x 256, K step 64) --------------------------------
//
// A tile [128][64] bf16: a row is 128 bytes = 8 chunks, so slot = chunk.
//   8 consecutive rows, same chunk -> one slot 8 times: 8-way conflict.
//   Stored chunk = chunk ^ (row & 7) -> 8 different slots. (v3's
//   (row >> 1) & 3 was for 64-byte rows; on 128-byte rows it leaves 2-way
//   conflicts, which the emulator's negative control shows.)
// B tile [64][256] bf16: a row is 512 bytes = 32 chunks, slot = chunk % 8.
//   Same fix as before: chunk ^ (row & 7) flips the low 3 bits per row.
namespace sparsebench::tile_v5 {

constexpr int kTileN = 256;
constexpr int kTileK = 64;
constexpr int kThreads = 256;  // 8 warps, 2 x 4
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kWarpN = kTileN / kWarpsN;  // 64 columns per warp
constexpr int kMmaN = kWarpN / 8;         // 8 mma tiles across
constexpr int kChunk = 8;                 // bf16 per 16-byte chunk

__device__ __forceinline__ int a_offset(int row, int chunk) {
  return row * kTileK + ((chunk ^ (row & 7)) * kChunk);
}
__device__ __forceinline__ int b_offset(int row, int chunk) {
  return row * kTileN + ((chunk ^ (row & 7)) * kChunk);
}

}  // namespace sparsebench::tile_v5
