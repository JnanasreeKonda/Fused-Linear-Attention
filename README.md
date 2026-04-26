# FusedLinearAttention

FusedLinearAttention is a course project on fusing QKV projection and scaled
dot-product attention into a single CUDA kernel for transformer inference.
The repository is organized around the pieces that are actually implemented:
the baseline ETTh1 pipeline, the fused-kernel code path, and the correctness
and profiling utilities used to compare them.

## Repository Layout

### `baseline_pipeline/`
End-to-end model, data, profiling, and results code.

- ETTh1 preprocessing and DataLoaders
- PatchTST baseline model
- baseline training and evaluation
- baseline and fused benchmark scripts
- canonical results, plots, and Phase 1 deliverables

### `kernel/`
Canonical fused-kernel implementation and loader.

- CUDA kernel source
- PyTorch C++ extension binding
- kernel loading logic
- tiling and shared-memory design notes

### `tests/`
Canonical root-level correctness checks.

- NumPy oracle used by the consolidated repo
- fused-kernel correctness suite

### `CPU_Reference_in_NumPy/`
Original NumPy reference implementation and golden outputs.

- reference implementation used to generate trusted outputs
- PyTorch comparison test
- saved golden `.npy` artifacts

### `data/`
Dataset storage.

- `ETTh1.csv`

## Current Implementation Status

Implemented in the repo:

- baseline ETTh1 pipeline and PatchTST training/evaluation
- baseline profiling workflow
- canonical fused-kernel source, binding, and benchmark scaffold
- canonical correctness suite and NumPy oracle
- merged results/figure generation scripts

Still partial or pending:

- real GPU validation of the fused kernel in this cleaned layout
- full end-to-end fused PatchTST training run
- support for model-side `d_head=32` in the current fused integration path

## Team Contributions

### Jnanasree Konda

- NumPy fused-attention reference workflow and golden-output generation
- correctness-oriented testing artifacts and validation logic
- PyTorch extension binding interface for the fused kernel

### Bhanuja Karumuru

- kernel tiling strategy and hardware-efficiency design
- fused CUDA kernel implementation in the canonical `kernel/` path
- fused profiling, comparison-table, and figure-generation scaffolding

### Rithwik Amajala

- ETTh1 preprocessing pipeline and dataset handling
- PatchTST baseline model, training loop, and evaluation workflow
- baseline profiling workflow and model-side fused wrapper/integration scaffold

## Notes

- `main` now uses the canonical layout above.
- Folder-level READMEs describe each component in more detail.
- Local scratch outputs at the repo root should not be treated as official
  deliverables; canonical outputs belong under `baseline_pipeline/results/`.
