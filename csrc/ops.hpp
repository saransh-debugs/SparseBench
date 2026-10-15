#pragma once

#include <torch/extension.h>

namespace sparsebench {

// Returns an int32 tensor of length n. Every element holds __CUDA_ARCH__
// of the GPU code that actually ran (800 on A100, 900 on H100).
// Proves the build shipped code for this GPU, not just PTX that gets
// recompiled at load time.
torch::Tensor arch_probe(int64_t n);

}  // namespace sparsebench
