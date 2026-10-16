import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 8.0 = A100 (Colab). 9.0a = H100 (SJSU).
# The "a" matters: wgmma and TMA only exist in sm_90a, not plain sm_90.
# Set TORCH_CUDA_ARCH_LIST yourself to build one arch and halve compile time.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0a")

setup(
    ext_modules=[
        CUDAExtension(
            name="sparsebench._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/smoke.cu",
                "csrc/routing.cu",
                "csrc/dsd.cpp",
                "csrc/dsd_v1.cu",
                "csrc/dsd_v2.cu",
                "csrc/dsd_v3.cu",
                "csrc/dsd_v4.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # -lineinfo lets ncu map stalls back to source lines.
                # It does not slow the kernel down.
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
