#include <torch/extension.h>

#include "ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arch_probe", &sparsebench::arch_probe,
        "Fill a tensor with the __CUDA_ARCH__ of the code that ran", py::arg("n"));
}
