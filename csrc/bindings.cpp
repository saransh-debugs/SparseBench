#include <torch/extension.h>

#include "ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arch_probe", &sparsebench::arch_probe,
        "Fill a tensor with the __CUDA_ARCH__ of the code that ran", py::arg("n"));
  m.def("build_topology_cuda", &sparsebench::build_topology_cuda,
        "Routing -> block-sparse layout on the GPU, no host sync", py::arg("expert_idx"),
        py::arg("num_experts"), py::arg("ffn_dim"), py::arg("block_size"));
  m.def("dsd_cuda", &sparsebench::dsd_cuda, "Block-sparse (BSR) @ dense on Tensor Cores", py::arg("blocks"),
        py::arg("w"), py::arg("row_offsets"), py::arg("col_indices"), py::arg("version"));
  m.def("sdd_cuda", &sparsebench::sdd_cuda, "Dense @ dense into the block-sparse layout, with fused GELU",
        py::arg("x"), py::arg("w1"), py::arg("row_token"), py::arg("row_block_expert"), py::arg("row_offsets"),
        py::arg("num_nnz"), py::arg("apply_gelu"));
}
