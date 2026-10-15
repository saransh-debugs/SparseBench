import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_kernel_runs_native_code_for_this_gpu() -> None:
    from sparsebench import _C

    # 1000 is not a multiple of the 256-thread block, so the bounds check runs.
    out = _C.arch_probe(1000)
    torch.cuda.synchronize()

    major, minor = torch.cuda.get_device_capability()
    assert out.shape == (1000,)
    assert torch.all(out == major * 100 + minor * 10).item()


def test_rejects_bad_size() -> None:
    from sparsebench import _C

    with pytest.raises(RuntimeError, match="must be positive"):
        _C.arch_probe(0)
