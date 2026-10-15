#include <torch/extension.h>

#include "ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arch_probe", &sparsebench::arch_probe,
        "Fill a tensor with the __CUDA_ARCH__ of the code that ran", py::arg("n"));
  m.def("build_topology_cuda", &sparsebench::build_topology_cuda,
        "Routing -> block-sparse layout on the GPU, no host sync", py::arg("expert_idx"),
        py::arg("num_experts"), py::arg("ffn_dim"), py::arg("block_size"));
}
