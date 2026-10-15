"""Block-sparse MoE matmul kernels."""

import torch  # noqa: F401  (must load before _C, which links against libtorch)

from sparsebench import _C

__all__ = ["_C"]
