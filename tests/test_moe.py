import pytest
import torch

from sparsebench.reference import moe_reference, route
from tests.cases import routing_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def make_layer(num_tokens: int, num_experts: int, d_model: int, ffn_dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, d_model, generator=g).bfloat16().cuda()
    w1 = (torch.randn(num_experts, d_model, ffn_dim, generator=g) / d_model**0.5).bfloat16().cuda()
    w2 = (torch.randn(num_experts, ffn_dim, d_model, generator=g) / ffn_dim**0.5).bfloat16().cuda()
    return x, w1, w2


def check(x, idx, weight, w1, w2) -> None:
    from sparsebench.moe import moe_forward

    out = moe_forward(x, idx, weight, w1, w2)
    # fp32 math on the same bf16 values. Our H is rounded to bf16 between the
    # two matmuls (like any bf16 MoE), hence a slightly wider tolerance.
    expected = moe_reference(x.float(), idx, weight.float(), w1.float(), w2.float())
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("case", ["random", "zipf", "skewed", "empty_middle"])
@pytest.mark.parametrize(("num_experts", "d_model", "ffn_dim"), [(8, 256, 512), (64, 256, 256)], ids=["E8", "E64"])
def test_moe_layer_matches_reference(case: str, num_experts: int, d_model: int, ffn_dim: int) -> None:
    x, w1, w2 = make_layer(300, num_experts, d_model, ffn_dim)
    idx = routing_case(case, 300, num_experts).cuda()
    weight = torch.rand(300, 2, generator=torch.Generator().manual_seed(1)).bfloat16().cuda()
    check(x, idx, weight, w1, w2)


def test_moe_layer_with_router() -> None:
    x, w1, w2 = make_layer(500, 8, 256, 512)
    logits = x.float() @ torch.randn(256, 8, generator=torch.Generator().manual_seed(2)).cuda()
    idx, weight = route(logits)
    check(x, idx, weight.bfloat16(), w1, w2)


def test_moe_layer_cuda_graph_proves_no_host_sync() -> None:
    # Graph capture fails on any host sync. Replaying with new routing and new
    # tokens proves the graph reads its inputs instead of baking in a result.
    from sparsebench.moe import moe_forward

    num_tokens, num_experts, d_model, ffn_dim = 512, 16, 256, 256
    x, w1, w2 = make_layer(num_tokens, num_experts, d_model, ffn_dim)
    idx = routing_case("random", num_tokens, num_experts, seed=1).cuda()
    weight = torch.rand(num_tokens, 2).bfloat16().cuda()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside capture, as torch requires
        moe_forward(x, idx, weight, w1, w2)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = moe_forward(x, idx, weight, w1, w2)

    for seed, case in [(2, "zipf"), (3, "empty_middle")]:
        new_x, _, _ = make_layer(num_tokens, num_experts, d_model, ffn_dim, seed=seed)
        new_idx = routing_case(case, num_tokens, num_experts, seed=seed).cuda()
        x.copy_(new_x)
        idx.copy_(new_idx)
        graph.replay()
        torch.cuda.synchronize()
        expected = moe_reference(x.float(), idx, weight.float(), w1.float(), w2.float())
        torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


def test_stage_marks_fire_in_order_and_change_nothing() -> None:
    from sparsebench.moe import moe_forward

    x, w1, w2 = make_layer(300, 8, 256, 512)
    idx = routing_case("zipf", 300, 8).cuda()
    weight = torch.rand(300, 2, generator=torch.Generator().manual_seed(1)).bfloat16().cuda()
    seen: list[str] = []
    marked = moe_forward(x, idx, weight, w1, w2, mark=seen.append)
    assert seen == ["routing", "SDD + GELU (fused)", "DSD", "combine"]
    assert torch.equal(marked, moe_forward(x, idx, weight, w1, w2))
