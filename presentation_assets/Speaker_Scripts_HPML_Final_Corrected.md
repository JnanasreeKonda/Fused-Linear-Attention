# FusedLinearAttention — Corrected Speaker Scripts
## HPML Final Presentation · Spring 2026
### 3 speakers · ~12 minutes total

---

## Consistency Fixes Applied

- Use the latest `origin/main` results from `baseline_pipeline/results`.
- CUDA correctness is reported as **11/11 passed, max abs diff <= 1.5e-7**.
- Runtime table uses the refreshed H100 fused timings: **872.0 us to 9124.8 us**.
- End-to-end PatchTST result is reported honestly: **fused MSE is +18.27% worse**, not within 1%.
- Kernel tile discussion uses the current kernel/report design: **T=64**, dynamic shared memory, `PROJ_K_TILE=8`, and grid `(B, H, q_tile)`.
- Main conclusion: fusion reduced memory traffic and kernel launches, but scalar fp32 projection loops still lose to PyTorch/cuBLAS/SDPA on H100.

---

## BHANUJA — Slides 1, 2, 3, 4
**Covers: intro, problem, challenges, kernel architecture.**
**Target: ~4 minutes**

### Slide 1 — Title

Good morning everyone. I’m Bhanuja, and together with Jnanasree and Rithwik, we built FusedLinearAttention: a custom CUDA kernel that fuses QKV projection and scaled dot-product attention into one GPU kernel.

The core idea is simple. Standard Transformer attention first computes Q, K, and V, writes those tensors to GPU global memory, and then launches another kernel to read them back and compute attention. We wanted to remove that intermediate HBM roundtrip by computing Q, K, and V inside the fused kernel and keeping those values in shared memory while attention is computed.

Our final project includes the CUDA kernel, PyTorch extension binding, correctness tests, profiling pipeline, and an end-to-end PatchTST validation path on ETTh1.

### Slide 2 — Executive Summary

The problem we target is memory movement between separate kernels. In the unfused path, the first kernel performs QKV projection, materializes Q, K, and V in HBM, and then the attention kernel reads those tensors back. That is extra memory traffic and an extra launch boundary.

Our solution is one fused CUDA kernel. It computes Q, K, and V, applies online tiled softmax, and writes only the final attention output back to HBM. The value delivered is real but nuanced. We reduce kernel launches from 2 to 1, and our estimated HBM reads drop by about 11% at sequence length 64 and up to about 55% at sequence length 1024.

Correctness is also strong: all 11 CUDA correctness configurations pass, with max absolute error at or below 1.5e-7 against the reference path.

But the honest result is that the current fused kernel is slower than the PyTorch baseline. The bottleneck is not correctness or memory traffic anymore. The bottleneck is compute efficiency: our projection path still uses scalar fp32 loops, while PyTorch uses highly optimized cuBLAS and SDPA kernels that can exploit H100 hardware much better.

### Slide 3 — Technical Challenges

We had four main challenges.

First was environment compatibility. The final compiled experiments needed a CUDA 12-compatible setup, so we ran the measured H100 kernel path on Chameleon Cloud and compiled for the active H100 architecture, `sm_90`.

Second was shared memory and occupancy. The kernel uses tiled shared memory for Q, K, and V, with padding to avoid bank conflicts. The final design uses tile size 64 for the benchmark kernel, giving about 48.8 KB of shared memory for the main Q/K/V staging arrays and about 65 KB when including the dynamic projection staging buffers. That keeps multiple blocks resident while still fitting the shared-memory budget.

Third was the dependency chain. Correctness had to be stable before profiling could mean anything. Once the reference and golden outputs were in place, we could safely optimize the kernel and know whether regressions were numerical or performance-related.

Fourth was numerical stability. A naive softmax can overflow at long sequence lengths. We use online softmax, which tracks a running max and denominator tile by tile. That lets us compute attention without storing the full score matrix and keeps the result numerically stable.

### Slide 4 — Kernel Architecture

The architecture is the main idea of the project. Before fusion, the baseline does QKV projection in one kernel, writes Q, K, and V to HBM, then launches attention as a second kernel.

After fusion, one CUDA kernel owns the full QKV-plus-attention path. The current launch structure uses a 3D grid over batch, head, and query tile. That was one of the most important optimizations, because the original one-block-per-batch-head design did not expose enough parallelism on H100.

Inside each block, threads cooperate through shared memory. We use a padded stride of `d + 1`, so for `d=64` the stride is 65. That avoids the classic 32-way shared-memory bank conflict. The attention computation uses a tiled loop over K/V blocks and online softmax.

The big takeaway from my section is: the fused architecture works and reduces memory movement. The remaining gap is that the math inside the projection stage is not yet tensor-core friendly. I’ll hand it over to Jnanasree to walk through implementation and measured results.

---

## JNANASREE — Slides 5, 6, 7, 8
**Covers: implementation, results, baseline profiling, optimization analysis.**
**Target: ~4 minutes**

### Slide 5 — Implementation Details

Thanks Bhanuja. I’m Jnanasree, and I’ll cover what we actually implemented and what the latest results show.

The CUDA kernel is in `kernel/fused_attn.cu`. The PyTorch C++ binding is in `kernel/fused_attn_ext.cpp`, and the JIT loader is in `kernel/load_kernel.py`. The profiling scripts are under `baseline_pipeline/profiling`, and the canonical result CSVs are under `baseline_pipeline/results`.

My main contribution was the correctness infrastructure. We built a NumPy reference implementation that mirrors the attention math directly: `Q = XWq`, `K = XWk`, `V = XWv`, then `softmax(QK^T / sqrt(d))V`. That reference is intentionally simple and auditable. We also generated deterministic golden outputs across batch sizes and sequence lengths, and used them to test the CUDA kernel.

This matters because kernel optimization can be misleading. A faster kernel that silently changes the math is not useful. Our correctness suite let us separate two questions: “Is the kernel computing the right answer?” and “Is it fast enough?”

### Slide 6 — Main Results

Here are the latest H100 compiled-kernel results.

The PyTorch baseline is very fast: about 110 microseconds for sequence lengths 64 through 256, about 119 microseconds at 512, and about 199 microseconds at 1024.

Our fused kernel is slower across the board. It takes about 872 microseconds at sequence length 64, 1422 microseconds at 128, 2533 microseconds at 256, 4737 microseconds at 512, and 9125 microseconds at 1024.

So the speedup column is below 1. At N=64, we are at about 0.13x baseline speed. At N=1024, we are at about 0.02x baseline speed.

But the memory result goes in the right direction. Estimated HBM reads fall by 11% to 55% as sequence length grows, and kernel launches drop from 2 to 1. That means the fusion hypothesis was directionally right for memory traffic. The runtime loss comes from compute inefficiency, mainly scalar fp32 projection loops.

### Slide 7 — Baseline Profiling

The baseline is hard to beat because it is not a naive implementation. PyTorch routes projection through optimized matrix multiplication and attention through optimized SDPA kernels. On H100, those paths are extremely tuned and can make use of the hardware much better than our scalar loop implementation.

The baseline timing is also fairly flat from N=64 to N=512, around 109 to 119 microseconds. At N=1024 it rises to about 199 microseconds, which is where the quadratic attention cost becomes more visible.

This is why our result is still valuable even though it is slower. It tells us exactly what kind of optimization is needed next. Memory movement is improved, but the projection compute path has to be rewritten around tensor-core-friendly tiles.

### Slide 8 — Speedup Analysis

We tried nine optimization passes. The most important improvement was changing the launch structure from one block per batch-head pair to one block per batch-head-query-tile. That exposed much more parallelism on H100.

Architecture-aware compilation for `sm_90` was necessary for the H100 runs. Dynamic shared memory let us support the current staging layout. Projection tiling through shared memory was another major improvement, because projection was the dominant cost.

Some ideas did not help. Two-pass softmax reduced score storage but added too much recomputation. Register cap sweeps made the compiler’s choices worse. A cooperative query-group rewrite preserved correctness but added synchronization and staging overhead, so it regressed performance and was reverted.

The final diagnosis is clean: the kernel is correct and memory-efficient, but not compute-efficient. The next real step is to use fp16 or bf16 and rewrite projection around WMMA or MMA tensor-core tiles. That is the path from a correct research kernel toward a competitive performance kernel.

Over to Rithwik for model validation, conclusions, and future work.

---

## RITHWIK — Slides 9, 10, 11, 12
**Covers: tiling/memory analysis, correctness validation, model results, conclusion.**
**Target: ~4 minutes**

### Slide 9 — Tiling & Memory Analysis

Thanks Jnanasree. I’m Rithwik, and I’ll connect the kernel-level results to validation and what we learned.

The tiling analysis is about fitting useful work into shared memory while keeping enough resident blocks per SM. We considered tile sizes 16, 32, 64, and 128. Larger tiles increase reuse but consume more shared memory. Smaller tiles improve occupancy but may not provide enough useful work per block.

The current kernel design uses tile size 64. For the Q, K, and V shared arrays, that is about 48.8 KB per block, and with the projection staging buffers the dynamic shared memory allocation is about 65 KB. Tile 128 is much tighter and reduces residency, while smaller tiles expose less work per block.

The bank-conflict padding is a small but important detail. With a raw stride of 64, threads in a warp can collide on the same shared-memory bank. Padding the stride to 65 changes the bank mapping so adjacent threads hit different banks. That removes a serialization point at negligible memory cost.

### Slide 10 — Correctness Validation

Our validation framework has three levels.

Level one is the NumPy mathematical reference. It is slow, but it is clear and easy to audit.

Level two is deterministic golden outputs. Those give us repeatable expected outputs for known configurations.

Level three is the actual CUDA kernel correctness suite. In the latest results, all 11 CUDA configurations pass. The tested shapes include sequence lengths from 64 to 1024, batch sizes 1 and 4, model dimensions 128 and 512, and up to 8 heads. The maximum absolute difference is at or below 1.5e-7, which is well within the fp32 tolerance we care about.

For end-to-end validation, we integrated the fused attention path into PatchTST on ETTh1. The baseline test result is MSE 180.46 and MAE 12.65. The fused result is MSE 213.44 and MAE 13.51. That is about 18.27% worse MSE and 6.82% worse MAE, so it does not meet the less-than-1% degradation target.

That tells us the fused path is functionally integrated, but not yet a drop-in replacement for model quality and speed.

### Slide 11 — Conclusion & Future Work

The project delivered a working custom CUDA fused-attention kernel, a PyTorch binding, a repeatable correctness framework, H100 compiled-kernel measurements, and end-to-end PatchTST validation.

What worked: fusion reduced kernel launches from 2 to 1, reduced estimated HBM reads by 11% to 55%, and passed all correctness tests.

What did not work yet: wall-clock speedup and model quality parity. The fused forward is still slower than baseline, and the final PatchTST fused metrics are worse than the unfused baseline.

The main lesson is that reducing memory traffic is necessary but not sufficient. On H100, compute structure matters enormously. If projection is written as scalar fp32 loops, the kernel cannot compete with PyTorch’s optimized tensor-core-backed path.

The highest-return future work is clear: rewrite projection using tensor-core MMA tiles, move to fp16 or bf16, improve warp-level work partitioning, and implement a true fused CUDA backward kernel instead of relying on the current recomputation bridge or fallback behavior.

### Slide 12 — Thank You

To close: this project is an honest systems result. We did not get a wall-clock speedup over PyTorch, but we did build the full path needed to understand why.

The kernel is correct. The memory-traffic reduction is real. The profiling pipeline is repeatable. And the bottleneck is now specific: tensor-core projection and a true optimized backward path.

Thank you. We’re happy to take questions.

---

## Timing Guide

| Speaker | Slides | Budget |
| --- | --- | --- |
| Bhanuja | 1–4 | ~4 min |
| Jnanasree | 5–8 | ~4 min |
| Rithwik | 9–12 | ~4 min |
| Total | 12 slides | ~12 min |

## Likely Q&A Owners

| Question | Owner |
| --- | --- |
| Why is the fused kernel slower? | Jnanasree |
| What helped most? | Bhanuja or Jnanasree |
| What did not help? | Jnanasree |
| How did you validate correctness? | Jnanasree |
| What does HBM reduction mean? | Bhanuja |
| Why did PatchTST quality degrade? | Rithwik |
| What is the next optimization step? | Rithwik or Jnanasree |
