"""Block-sparse MoE matmul kernels.

The CUDA extension is not imported here. Import it as `from sparsebench import _C`
where needed, so the pure-PyTorch modules still work on machines without CUDA.
"""
