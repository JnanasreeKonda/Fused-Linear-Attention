# FusedLinearAttention

FusedLinearAttention is an HPML course project on reducing transformer
attention overhead on GPU. The current best implementation in this repository
is a custom warp-cooperative attention kernel that works with optimized
PyTorch projection on H100.

The repo now contains:

- a complete ETTh1 + PatchTST baseline pipeline
- custom CUDA attention backends under `kernel/`
- correctness, profiling, validation, and figure-generation workflows
- result artifacts under
  [baseline_pipeline/results](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results)

The short version: the current fastest custom path in this repo is the
`hybrid_warp4` backend, which keeps projection on the optimized PyTorch side
and uses our warp-cooperative CUDA kernel for the attention step.

## Project Goal

The unfused attention path does this in two stages:

1. project input tokens into `Q`, `K`, and `V`
2. read those intermediates back to run attention

The project aims to:

- reduce kernel launches
- reduce global-memory traffic
- reduce costly memory movement through HBM

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
- H100 custom-kernel vs baseline profiling run
- H100 mixed-precision hybrid and warp-cooperative tuning runs

### What is still not ideal

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

### 1. Final custom backend vs baseline

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

### 2. Forecast metrics

| Model | MSE | MAE |
| --- | ---: | ---: |
| Baseline | 180.46336365 | 12.64666843 |
| Custom attention path | 213.58927917 | 13.51428318 |

Delta vs baseline:

- MSE: `+18.35603352%`
- MAE: `+6.86042142%`

### 3. End-to-end timing

| Model | Mean Epoch Time (`s`) | Forward / iter (`ms`) |
| --- | ---: | ---: |
| Baseline | 1.8825 | 0.516714 |
| Custom attention path | 2.5510 | 0.777603 |

### 4. Correctness

- `11 / 11` CUDA correctness cases passed
- max abs diff stayed on the order of `1e-7`

## Memory Story

The final fastest kernel is a hybrid design. It still uses normal projection,
but it lowers the analytic HBM footprint mainly through `bf16` storage for the
input, weights, and projected tensors.

For the benchmark shape used here, the final hybrid kernel's analytic memory
breakdown is:

- estimated HBM reads: about `50%` lower than the fp32 baseline
- estimated HBM writes: about `37.5%` lower than the fp32 baseline

These newer graph assets are included in
[baseline_pipeline/results/figures](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures):

- [baseline_vs_final_time.svg](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/baseline_vs_final_time.svg)
- [baseline_vs_final_memory_breakdown.svg](/Users/jnanasreekonda/PycharmProjects/Fused-Linear-Attention/baseline_pipeline/results/figures/baseline_vs_final_memory_breakdown.svg)

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

## Why The Final Kernel Is Still Slower At Larger Sequence Lengths

Even after the successful optimizations, the current custom kernel still loses to
the PyTorch baseline at larger sequence lengths because:

- the baseline still uses extremely optimized library kernels end to end
- the custom attention stage is much better than before, but still not as
  hardware-efficient at `N>=256`
- the model path still uses a recomputation-based backward bridge during training

The final kernel is therefore:

- **correct**
- **more memory-efficient on paper**
- **not yet consistently compute-efficient enough to win on H100**

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

- CUDA kernel implementation
- Mixed-precision templates
- Correctness suites (Levels 2–3)

### Bhanuja Karumuru

- Tiling strategy
- Shared-memory design
- HBM model, NumPy oracle
- WMMA benchmarking

### Rithwik Amajala

- PyTorch integration
- Autograd bridge
- Backend dispatch
- Microbenchmark harness
