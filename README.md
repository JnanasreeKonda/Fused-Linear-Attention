# FusedLinearAttention

FusedLinearAttention is a course project on reducing transformer inference
overhead by fusing QKV projection and scaled dot-product attention into a
single CUDA kernel. The repository now combines three major pieces:

- a working ETTh1 + PatchTST baseline pipeline
- a canonical custom CUDA kernel and PyTorch extension path
- correctness, profiling, validation, and figure-generation workflows

## Project Goal

The project targets the standard two-stage attention path:

1. separate Q, K, and V projections that write intermediate tensors to HBM
2. attention that reads those tensors back from HBM

The fused kernel path aims to reduce kernel launches and external memory
traffic by combining these stages into a single implementation.

## Repository Structure

### `baseline_pipeline/`

- ETTh1 preprocessing and DataLoaders
- PatchTST baseline model
- baseline training and evaluation scripts
- baseline and fused benchmark entrypoints
- Phase 3 validation scripts
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
| Fused profiling pipeline | Complete | Fused benchmark, CSV merge, occupancy sweep, and figure generation are implemented. |
| Phase 3 validation pipeline | Complete | End-to-end validation, checkpoint conversion, timing, and comparison-table generation are implemented. |
| Real compiled-kernel GPU validation | Partial | Environment-dependent; simulation fallback works, but compiled-kernel runs require a compatible CUDA toolchain. |
| End-to-end fused PatchTST validation | Partial | The fused wrapper now supports CUDA forward plus a custom backward bridge, but final retraining/benchmarking still depends on a working compiled-kernel environment. |

## Remaining Work

- real GPU validation of the fused kernel in this cleaned layout
- full end-to-end fused PatchTST training run and final metric collection
- optional future work: replace the current custom autograd backward bridge
  with a handwritten fused CUDA backward kernel

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

Quick simulation:

```bash
python tests/test_correctness.py --simulate --quick --use-root-oracle
python tests/test_fused_backward.py
```

Quick CUDA-backed check:

```bash
python tests/test_correctness.py --quick
```

Full suite:

```bash
python tests/test_correctness.py
```

### Phase 2 / 3 helper flow

```bash
cd baseline_pipeline
./run_phase23.sh
```

### Fused profiling pipeline

Simulation:

```bash
cd baseline_pipeline
./run_bhanuja.sh --simulate
```

CUDA-backed:

```bash
cd baseline_pipeline
./run_bhanuja.sh
```

### Phase 3 validation

CPU/reference-path validation:

```bash
cd baseline_pipeline
./run_phase3.sh --no-cuda
```

CUDA-backed fused retraining and validation:

```bash
cd baseline_pipeline
./run_phase3.sh --train-fused
```

Direct fused training entrypoint:

```bash
cd baseline_pipeline
python model/train.py --attention fused
```

### Full repo helper

```bash
./run_all.sh
```

## Results

Canonical outputs live under `baseline_pipeline/results/`.

Important files:

- `correctness_results.csv`
- `baseline_profiling.csv`
- `fused_profiling.csv`
- `comparison_table.csv`
- `occupancy_sweep.csv`
- `validation_table.csv`
- `endtoend_timing.csv`
- `fused_model_metrics.csv`
- `figures/`

The fused profiling CSV includes comparison metadata so simulation-mode runs
and real CUDA-kernel runs are distinguishable in the repo and in the final
report.

## Limitations

- Final fused-kernel performance claims should only be made from directly
  comparable CUDA-backed runs.
- The custom CUDA extension can still fail to build on shared systems with
  older toolkit / host compiler combinations.
- Fully validated fused retraining still depends on a CUDA environment where
  the compiled kernel runs successfully end to end.
- The current backward implementation is a custom autograd bridge that
  recomputes the PyTorch reference attention graph during backpropagation. It
  enables training on the fused forward path, but it is not yet a fully fused
  handwritten CUDA backward kernel.

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
- baseline profiling workflow and model-side fused wrapper / validation scaffold

## Notes

- `main` uses the canonical repo layout.
- Folder-level READMEs describe each component in more detail.
- Canonical deliverables belong under `baseline_pipeline/results/`.
