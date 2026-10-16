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
