# FusedLinearAttention

FusedLinearAttention is a course project on reducing transformer inference
overhead by fusing QKV projection and scaled dot-product attention into a
single CUDA kernel. The repository contains three major pieces:

- a working ETTh1 + PatchTST baseline pipeline
- a canonical custom CUDA kernel and PyTorch extension path
- correctness, benchmarking, and figure-generation utilities used to compare
  the fused path against the unfused baseline

## Project Goal

The project targets the standard two-stage attention path:

1. separate Q, K, and V projections that write intermediate tensors to HBM
2. attention that reads those tensors back from HBM

The fused kernel path aims to reduce kernel launches and external memory
traffic by combining these stages into a single implementation for the
benchmark configuration.

## Repository Structure

### `baseline_pipeline/`

- ETTh1 preprocessing and DataLoaders
- PatchTST baseline model
- baseline training and evaluation scripts
- baseline and fused benchmark entrypoints
- result-merging and figure-generation helpers

### `kernel/`

- canonical CUDA kernel source
- PyTorch C++ extension binding
- JIT loader
- tiling and shared-memory design notes

### `tests/`

- canonical correctness checks
- NumPy-backed reference comparison

### `CPU_Reference_in_NumPy/`

- original NumPy oracle
- PyTorch-vs-NumPy comparison test
- golden `.npy` outputs

### `data/`

- `ETTh1.csv`

## Milestones And Status

| Milestone | Status | Notes |
| --- | --- | --- |
| Dataset setup and ETTh1 preprocessing | Complete | Canonical ETTh1 pipeline is in `baseline_pipeline/model/data.py`. |
| Baseline PatchTST model | Complete | Training and evaluation flow is implemented. |
| Baseline profiling | Complete | `baseline_pipeline/profiling/baseline_bench.py` produces the unfused benchmark CSV. |
| NumPy oracle and golden outputs | Complete | Reference implementation and saved outputs are present. |
| Canonical kernel source and binding | Complete | CUDA source, C++ binding, and loader are in `kernel/`. |
| Root correctness workflow | Complete | Root `tests/` path is the source of truth for correctness checks. |
| Fused profiling pipeline | Complete | Fused benchmark, CSV merge, and figure generation are implemented. |
| Real compiled-kernel GPU validation | Partial | Environment-dependent; simulation fallback works, but compiled-kernel runs require a compatible CUDA toolchain. |
| End-to-end fused PatchTST validation | Partial | The current fused path remains specialized and is not yet a complete drop-in replacement for all model settings. |

## How To Run

### Baseline pipeline

```bash
cd baseline_pipeline
python model/data.py
python profiling/baseline_bench.py
python model/train.py
python model/evaluate.py
```

### Correctness

```bash
python tests/test_correctness.py
python tests/test_correctness.py --simulate
```

### Fused profiling pipeline

```bash
cd baseline_pipeline
./run_bhanuja.sh --simulate
```

On a machine where the CUDA extension compiles successfully:

```bash
cd baseline_pipeline
./run_bhanuja.sh
```

## Results

Canonical outputs live under `baseline_pipeline/results/`.

Important files:

- `baseline_profiling.csv`
- `fused_profiling.csv`
- `comparison_table.csv`
- `occupancy_sweep.csv`
- `figures/hbm_bandwidth.png`
- `figures/speedup.png`
- `figures/occupancy_vs_tile.png`
- `figures/kernel_count.png`

The fused profiling CSV includes a `run_mode` column so simulation-mode runs
and real CUDA-kernel runs are distinguishable in the repo and in the final
report.

## Observations

- The baseline ETTh1 and PatchTST pipeline is runnable and benchmarkable from
  the cleaned repository layout.
- The fused benchmarking workflow is automated end to end: benchmark, merge,
  and figure generation all run from canonical paths.
- The correctness framework is also automated and can fall back to simulation
  when the compiled extension is unavailable.
- Final fused-kernel performance claims should only be made from runs where
  `run_mode=cuda_kernel`. Simulation-mode outputs validate the workflow, but
  they are not substitutes for compiled-kernel measurements.

## Limitations

- The kernel is currently specialized for the benchmark path with
  `HEAD_DIM=64`.
- Full end-to-end fused PatchTST support remains partial because the baseline
  model configuration uses a different head dimension.
- On some shared GPU environments, the custom CUDA extension may fail to build
  due to CUDA toolkit and host compiler incompatibilities. The repo preserves a
  simulation path so correctness and reporting workflows can still be exercised.

## Team Contributions

### Jnanasree Konda

- NumPy fused-attention reference workflow and golden-output generation
- correctness-oriented validation logic
- PyTorch extension binding interface for the fused kernel

### Bhanuja Karumuru

- fused kernel tiling strategy and hardware-efficiency design
- fused CUDA kernel implementation in the canonical `kernel/` path
- profiling pipeline, comparison-table merge, and figure-generation workflow

### Rithwik Amajala

- ETTh1 preprocessing pipeline and dataset handling
- PatchTST baseline model, training loop, and evaluation workflow
- baseline profiling workflow and model-side fused wrapper scaffold

## Notes

- `main` uses the canonical repo layout.
- Folder-level READMEs describe each component in more detail.
- Canonical deliverables belong under `baseline_pipeline/results/`.
