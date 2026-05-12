# FusedLinearAttention

FusedLinearAttention is an HPML course project on reducing transformer
attention overhead on GPU. We started with the classic systems idea of fusing
Q/K/V projection and scaled dot-product attention into one CUDA kernel, then
iterated toward faster practical variants on H100.

The repo now contains:

- a complete ETTh1 + PatchTST baseline pipeline
- a canonical fully fused CUDA extension path under `kernel/`
- newer hybrid and warp-cooperative custom attention backends
- correctness, profiling, validation, and figure-generation workflows
- result artifacts under
  [baseline_pipeline/results](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results)

The short version: the original fully fused kernel is **correct** and
**reduces estimated HBM traffic**, but it is still slower than the PyTorch
baseline. The current fastest custom path in this repo is the
`hybrid_warp4` backend, which keeps projection on the optimized PyTorch side
and uses our warp-cooperative CUDA kernel for the attention step.

## Project Goal

The unfused attention path does this in two stages:

1. project input tokens into `Q`, `K`, and `V`
2. read those intermediates back to run attention

The fused kernel aims to:

- reduce kernel launches
- reduce global-memory traffic
- avoid materializing full `Q/K/V` tensors in HBM

## Current Status

### What is complete

- ETTh1 preprocessing and DataLoaders
- PatchTST baseline model, training, and evaluation
- baseline profiling workflow
- canonical NumPy / PyTorch correctness oracle
- canonical CUDA kernel source and PyTorch extension binding
- fused profiling pipeline
- Phase 3 end-to-end validation pipeline
- H100 compiled-kernel correctness run
- H100 fused-vs-baseline profiling run
- H100 mixed-precision hybrid and warp-cooperative tuning runs

### What is still not ideal

- the original fully fused kernel does **not** beat the unfused PyTorch
  baseline on H100
- the best custom backend beats baseline only at smaller sequence lengths
- end-to-end custom-attention model quality is still worse than baseline
- backward is still a custom autograd bridge, not a handwritten fused CUDA
  backward kernel

## Repository Layout

### [baseline_pipeline/](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline)

- ETTh1 preprocessing
- PatchTST baseline
- baseline and fused profiling scripts
- Phase 3 end-to-end validation
- result merge and figure generation

### [kernel/](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/kernel)

- `fused_attn.cu`
- `fused_attn_ext.cpp`
- `load_kernel.py`
- `attn_only.cu`
- `attn_only_warp4.cu`
- kernel design notes

### [tests/](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/tests)

- correctness checks
- reference comparisons
- fused backward smoke test

### [CPU_Reference_in_NumPy/](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/CPU_Reference_in_NumPy)

- original NumPy oracle
- golden outputs
- reference comparison utilities

## H100 Results

Source files:

- [comparison_table.csv](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/comparison_table.csv)
- [validation_table.csv](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/validation_table.csv)
- [endtoend_timing.csv](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/endtoend_timing.csv)
- [correctness_results.csv](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/correctness_results.csv)

### 1. Original fully fused kernel vs baseline

| Seq Len | Baseline (`us`) | Fused (`us`) | Fused vs Baseline |
| --- | ---: | ---: | ---: |
| 64 | 109.7448 | 872.0335 | 0.1258x |
| 128 | 109.1533 | 1422.1433 | 0.0768x |
| 256 | 108.9229 | 2532.7407 | 0.0430x |
| 512 | 118.5030 | 4737.3379 | 0.0250x |
| 1024 | 198.5365 | 9124.8203 | 0.0218x |

This is the result that best demonstrates the original fusion idea: lower
kernel count and lower estimated HBM traffic, but not enough compute
efficiency to beat PyTorch.

### 2. Best current custom backend vs baseline

Our fastest current path is:

- backend: `hybrid_warp4`
- kernel dtype: `bfloat16`
- query-group size: `2`

H100 forward microbenchmark:

| Seq Len | Baseline (`us`) | Final Custom Kernel (`us`) | Final vs Baseline |
| --- | ---: | ---: | ---: |
| 64 | 107.2 | 88.0 | 1.22x |
| 128 | 142.1 | 100.5 | 1.41x |
| 256 | 127.2 | 164.2 | 0.77x |
| 512 | 121.6 | 293.3 | 0.41x |
| 1024 | 200.7 | 556.4 | 0.36x |

So the latest kernel work materially improved the repo:

- it beats baseline at `N=64`
- it beats baseline at `N=128`
- it is still slower at `N>=256`
- it is dramatically faster than the original fully fused kernel

### 3. Forecast metrics

| Model | MSE | MAE |
| --- | ---: | ---: |
| Baseline | 180.46336365 | 12.64666843 |
| Custom attention path | 213.58927917 | 13.51428318 |

Delta vs baseline:

- MSE: `+18.35603352%`
- MAE: `+6.86042142%`

### 4. End-to-end timing

| Model | Mean Epoch Time (`s`) | Forward / iter (`ms`) |
| --- | ---: | ---: |
| Baseline | 1.8825 | 0.516714 |
| Custom attention path | 2.5510 | 0.777603 |

### 5. Correctness

- `11 / 11` CUDA correctness cases passed
- max abs diff stayed on the order of `1e-7`

## Memory Story

There are two different memory stories in this repo.

The original fully fused kernel gives the cleanest fusion argument because it
avoids materializing full `Q/K/V` tensors in HBM. That is why its estimated
HBM-read reduction grows from `10.71%` at `N=64` to `54.55%` at `N=1024`.

The final fastest kernel is a hybrid design, so it does **not** remove
`Q/K/V` materialization in the same way. Its lower analytic HBM footprint comes
mainly from using `bf16` for input, weights, and projected tensors.

For the benchmark shape used here, the final hybrid kernel's analytic memory
breakdown is:

- estimated HBM reads: about `50%` lower than the fp32 baseline
- estimated HBM writes: about `37.5%` lower than the fp32 baseline

These newer graph assets are included in
[baseline_pipeline/results/figures](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures):

- [baseline_vs_final_time.svg](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/baseline_vs_final_time.svg)
- [baseline_vs_final_memory_breakdown.svg](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/baseline_vs_final_memory_breakdown.svg)

## Generated Figures

The main charts are already in
[baseline_pipeline/results/figures](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures):

- [speedup.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/speedup.png)
- [wall_time_speedup.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/wall_time_speedup.png)
- [hbm_bandwidth.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/hbm_bandwidth.png)
- [kernel_count.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/kernel_count.png)
- [occupancy_vs_tile.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/occupancy_vs_tile.png)
- [occupancy_vs_tile_tradeoff.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/occupancy_vs_tile_tradeoff.png)
- [nsight_timeline_comparison.png](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/nsight_timeline_comparison.png)

There is also a detailed written report here:

- [baseline_pipeline/final_report.md](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/final_report.md)

## How To Run

### 1. Baseline pipeline

```bash
cd baseline_pipeline
python model/data.py
python profiling/baseline_bench.py
python model/train.py
python model/evaluate.py
```

### 2. Correctness

Quick CUDA check:

```bash
python tests/test_correctness.py --quick
```

Full CUDA check:

```bash
python tests/test_correctness.py
```

Simulation/reference mode:

```bash
python tests/test_correctness.py --simulate --quick --use-root-oracle
python tests/test_fused_backward.py
```

### 3. Fused profiling

```bash
cd baseline_pipeline
./run_bhanuja.sh
```

### 4. Phase 3 validation

```bash
cd baseline_pipeline
./run_phase3.sh --train-fused
```

To run the newer best custom backend in the model path:

```bash
cd baseline_pipeline
export FLA_ATTN_BACKEND=hybrid_warp4
export FLA_KERNEL_DTYPE=bfloat16
export FLA_Q_GROUP_SIZE=2
python model/end_to_end_validate.py --train-fused
```

### 5. Full helper

```bash
./run_all.sh
```

## Environment Notes

For compiled-kernel runs you need:

- CUDA toolkit with `nvcc`
- a CUDA-compatible PyTorch build
- `ninja`
- a GPU where `torch.cuda.is_available()` is true

The H100 run used:

- host CUDA toolkit
- H100 GPU
- `TORCH_CUDA_ARCH_LIST=9.0`
- compiled extension path via [kernel/load_kernel.py](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/kernel/load_kernel.py)

## Why The Original Fully Fused Kernel Is Still Slower

Even after the successful optimizations, the current fused kernel still loses to
the PyTorch baseline because:

- Q/K/V projection is still implemented as scalar fp32 loops inside the kernel
- the baseline uses highly optimized library kernels
- reduced HBM traffic is being outweighed by compute inefficiency

That kernel is therefore:

- **correct**
- **more memory-efficient on paper**
- **not yet compute-efficient enough to win on H100**

## Best Current Optimization State

The best current custom backend in the repo now includes:

- `hybrid_warp4` attention backend
- `bfloat16` inputs / weights / projected tensors
- warp-cooperative query processing
- tuned query-group size `2`
- vectorized row loads and inner-loop attention math

This backend is the one to use when you want the fastest current custom
implementation in this repository.

## Recommended Next Work

If this project is extended further, the highest-return next steps are:

1. profile the final hybrid kernel with real hardware counters using Nsight
2. continue warp-level tuning for the `hybrid_warp4` backend at `N>=256`
3. redesign the attention stage to be even more Tensor-Core-friendly
4. implement a true fused CUDA backward kernel

## Team Contributions

### Jnanasree Konda

- fused-attention NumPy oracle and correctness work
- PyTorch extension interface
- kernel validation and GPU experiment integration

### Bhanuja Karumuru

- fused kernel design and CUDA implementation
- profiling pipeline, result merge, and figure generation

### Rithwik Amajala

- ETTh1 pipeline
- PatchTST baseline, training, and evaluation
- Phase 3 model-side integration and validation workflow
