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
- model-side fused attention wrapper with a safe fallback path

Still partial or pending:

- real GPU validation of the fused kernel in this cleaned layout
- full end-to-end fused PatchTST training run using the compiled kernel path
- kernel generalization beyond the benchmark-oriented `d_head=64` fused path

## Phase 2 and Phase 3 Tasks Finished On This Branch

### Phase 2

- root-level NumPy oracle in `tests/reference.py`
- stacked-QKV oracle path to mirror a fused projection interface
- canonical correctness runner in `tests/test_correctness.py`
- quick correctness mode for fast smoke testing
- correctness CSV output directed into `baseline_pipeline/results/`
- canonical extension loader in `kernel/load_kernel.py`

### Phase 3

- model-side `FusedLinearAttentionBlock` wrapper in `baseline_pipeline/model/fused_attn_block.py`
- safe fallback to PyTorch SDPA during training, CPU runs, or unsupported head sizes
- compiled-kernel path for CUDA eval / no-grad execution when `d_head == 64`
- `baseline_pipeline/run_phase23.sh` helper script for the Phase 2 / 3 validation flow

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

## How To Run

### Baseline Phase 1 flow

```bash
cd baseline_pipeline
chmod +x run_phase1.sh
./run_phase1.sh
```

### Phase 2 and Phase 3 simulation flow

This path does not require the compiled CUDA kernel:

```bash
cd baseline_pipeline
chmod +x run_phase23.sh
./run_phase23.sh
```

### Root correctness checks

Quick simulation:

```bash
python3 tests/test_correctness.py --simulate --quick --use-root-oracle
```

Quick CUDA-backed run:

```bash
python3 tests/test_correctness.py --quick
```

Full CUDA-backed run:

```bash
python3 tests/test_correctness.py
```

Expected output:

```bash
baseline_pipeline/results/correctness_results.csv
```

### Fused benchmark flow

Simulation:

```bash
cd baseline_pipeline
python3 profiling/fused_bench.py --simulate
```

CUDA-backed:

```bash
cd baseline_pipeline
python3 profiling/fused_bench.py
```

### PatchTST with the fused wrapper

```python
from baseline_pipeline.model.patchtst import PatchTST
from baseline_pipeline.model.fused_attn_block import FusedLinearAttentionBlock

model = PatchTST(attn_block_class=FusedLinearAttentionBlock)
```

Behavior today:

- training: PyTorch fallback path
- eval on CUDA with `torch.no_grad()` and `d_head == 64`: compiled fused kernel path
