# SparseBench

Custom CUDA kernels for Mixture-of-Experts (MoE) layers, written as block-sparse matrix multiplies (the MegaBlocks formulation), from routing to output, with **no host sync anywhere in the layer**.

> **Status: work in progress.** The A100 kernels, tests and benchmarks are done. The H100 port and the full write-up are still to come; see [TODO](#todo). Numbers below are measured on one Colab A100 and may move a few percent between runs.

## What's here

| Piece | File | What it does |
|---|---|---|
| Routing | `csrc/routing.cu` | Sorts token copies by expert and builds the block-sparse layout on the GPU. A counting sort with warp-level ranking (`__match_any_sync`), so it's stable without atomics. Buffers are sized for the worst case, so the CPU never waits for the GPU. |
| SDD | `csrc/sdd.cu` | First MoE matmul, written straight into the block-sparse layout. Gathers tokens inside the load (no sorted copy of `x`), zero-fills padding rows for free, and fuses GELU into the output. |
| DSD v1 → v5 | `csrc/dsd_v*.cu` | Second MoE matmul. Each version fixes one bottleneck that `ncu` measured in the one before (see below). |
| Full layer | `sparsebench/moe.py` | Routing → SDD → DSD → combine. A CUDA-graph test proves it runs with no host sync. |
| References | `sparsebench/reference.py` | Plain PyTorch ground truth that every kernel is tested against. |

## DSD, one bottleneck at a time

A100, block size 128, 4096 tokens, top-2. TFLOPS count useful work only. Sources: `bench/bench_dsd.py`, `bench/results_ncu.md`.

| Version | Change | What `ncu` showed it fixing | Mixtral-like TFLOPS |
|---|---|---|---:|
| v1 | Tensor Cores (`mma.sync`), plain loads | baseline | 42 |
| v2 | `cp.async` pipeline | stall on global memory: 10.7 → 0.3 | 62 |
| v3 | XOR swizzle + `ldmatrix` | shared-memory bank conflicts: 145M → ~0 | 137 |
| v4 | Row-block-first tile order, skip dead blocks | GPU memory bytes read: halved | 147 |
| v5 | 128×256 tiles, K step 64, 64×64 warp tiles | Tensor Cores busy: ~54% → ~64% | 174 |

## Against real MoE implementations

Whole MoE layer, routing included (`bench/bench_moe.py`). Time in ms, lower is better.

| | Fine-grained (64 experts) | Mixtral-like (8 experts) |
|---|---:|---:|
| **Ours** | **1.85** | 12.1 |
| Grouped GEMM (`torch._grouped_mm`) | 3.17 | **9.8** |
| MegaBlocks-style (`stk`) | 2.60 | 13.0 |

- **Many small experts:** ours is 1.4x faster than MegaBlocks-style and 1.7x faster than grouped GEMM, with no host sync.
- **Few big experts:** grouped GEMM wins by about 1.2x. With 8 big matmuls there's little sparsity left to exploit. Ours still edges out MegaBlocks-style.
- Our DSD kernel alone only ties MegaBlocks' (`bench/results_baselines.md`). The end-to-end win comes from the rest of the layer: the fused SDD, GPU-side routing, and no CPU round trip.

Uniform routing shown; Zipf-skewed routing gives the same picture. Full tables and run-to-run noise: `bench/results_moe.md`.

## Running it

Needs an NVIDIA GPU with compute capability 8.0+ (A100 / H100). The kernels are developed on Colab; `notebooks/colab_runner.ipynb` clones, builds, tests and benchmarks in one go.

```bash
TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -e ".[test]"
python -m pytest -q
python bench/bench_moe.py
```

The pure-PyTorch parts (`reference.py`, `topology.py`) and their tests also run on CPU.

## TODO

- [x] Re-run `bench_baselines.py` and `bench_moe.py` with the `stk` baseline on its fastest correct index widths (`bench/results_baselines.md`, `bench/results_moe.md`).
- [ ] Full sanitizer run over every kernel version (`FULL = True` in the notebook). Day-to-day runs only check the newest two.
- [ ] H100 port (`sm_90a`): `wgmma` + TMA + `mbarrier` pipeline, on SJSU's HPC. Same tests and benchmarks, results in their own column. First check that `ncu` works there.
- [ ] Close the Mixtral-like gap to grouped GEMM (~1.25x), or document clearly why block-sparse isn't the right tool for few, large experts.
- [ ] Per-stage time breakdown of the full layer (routing / SDD / DSD / combine) for ours and the `stk` pipeline, to show where the end-to-end win comes from.
- [ ] Full README write-up: design, every measured number with its source, limits, and how to reproduce.
