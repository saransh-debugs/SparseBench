# Token rounding vs plain top-k

From `bench/bench_token_rounding.py`. A100-SXM4-40GB (Colab), torch
2.11.0+cu130, top-2, tile 128. Recorded 2026-10-17. Token rounding = SonicMoE
(Guo et al., ICLR 2026), Algorithm 4: round each expert's top-k count to the
nearest multiple of 128. Every result was checked against an fp32 reference
on the same routing.

## Whole layer (ms), same router scores

| shape | tokens | routing | tiles / expert | ours top-k | ours TR | ours gain | grouped GEMM top-k | grouped GEMM TR | GEMM gain | rounding itself |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fine-grained | 4096 | uniform | 1 | 2.174 | 1.705 | 1.28x | 3.169 | 2.752 | 1.15x | 0.640 |
| fine-grained | 4096 | Zipf | 1 | 1.794 | 1.384 | 1.30x | 2.319 | 1.432 | 1.62x | 0.646 |
| fine-grained | 16384 | uniform | 4 | 4.794 | 4.463 | 1.07x | 5.726 | 5.718 | 1.00x | 0.842 |
| fine-grained | 16384 | Zipf | 4 | 4.934 | 4.436 | 1.11x | 5.339 | 4.342 | 1.23x | 0.831 |
| Mixtral-like | 4096 | uniform | 8 | 12.444 | 11.681 | 1.07x | 9.879 | 9.675 | 1.02x | 0.628 |
| Mixtral-like | 4096 | Zipf | 8 | 12.524 | 11.343 | 1.10x | 9.880 | 9.294 | 1.06x | 0.620 |

## How much routing changed (% of top-k picks, unless noted)

| shape | tokens | routing | top-k padding (% of rows) | picks dropped | assignments added | tokens with 0 experts (% of tokens) | tokens with > k experts | top-k score kept |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fine-grained | 4096 | uniform | 35.4 | 3.4 | 3.4 | 0.34 | 6.5 | 98.1 |
| fine-grained | 4096 | Zipf | 35.4 | 4.9 | 1.8 | 0.12 | 3.4 | 97.8 |
| fine-grained | 16384 | uniform | 10.5 | 1.5 | 1.9 | 0.10 | 3.6 | 99.2 |
| fine-grained | 16384 | Zipf | 13.2 | 1.9 | 1.1 | 0.02 | 2.2 | 99.3 |
| Mixtral-like | 4096 | uniform | 7.2 | 1.1 | 1.1 | 0.00 | 2.1 | 99.6 |
| Mixtral-like | 4096 | Zipf | 5.9 | 3.7 | 0.6 | 0.00 | 1.2 | 99.1 |

## What it shows

1. **The gain follows the padding.** 1.28-1.30x where top-k padding is 35%
   (1 tile per expert), 1.07-1.11x where it's 6-13%.
2. **The big gain is in the risky regime.** SonicMoE reports token rounding
   keeps quality when experts average at least 2 tiles. At 1 tile it changes
   3-5% of picks; in the safe regime (4 tiles) our gain is 7-11%.
3. **It helps grouped GEMM too**, sometimes more (Zipf, 1 tile: 1.62x), since
   grouped GEMM also wastes the last partial tile of every expert. With token
   rounding + Zipf at 1 tile, grouped GEMM nearly ties us (1.43 vs 1.38 ms);
   with uniform routing we stay 1.6x ahead.
4. **Our rounding step costs more than it saves.** The PyTorch implementation
   takes 0.62-0.84 ms (several small kernels plus one host sync), about the
   same at every size, so it's fixed overhead, not math. At fine-grained 4096
   that cancels the 0.47 ms gain. To pay off it would have to be fused into
   the routing kernel, which already counts tokens per expert.

Note: these routings come from `router_scores` (full softmax scores), not
`routing_case`, so the top-k times here aren't comparable to `bench_moe.py`.
