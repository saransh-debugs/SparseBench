#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>

#include "ops.hpp"

namespace sparsebench {
namespace {

__global__ void arch_probe_kernel(int32_t* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
#ifdef __CUDA_ARCH__
    out[i] = __CUDA_ARCH__;
#endif
  }
}

}  // namespace

torch::Tensor arch_probe(int64_t n) {
  TORCH_CHECK(n > 0, "arch_probe: n must be positive, got ", n);

  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(),
              "arch_probe: n too large for a 1D grid");

  auto out = torch::empty({n}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  // Launch on torch's current stream, not the default stream. The default
  // stream would silently sync with all of torch's work and hide ordering bugs.
  arch_probe_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int32_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace sparsebench
