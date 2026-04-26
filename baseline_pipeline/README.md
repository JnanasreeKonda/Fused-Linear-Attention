# `baseline_pipeline/`

This folder contains the canonical model-side workflow for the project:
dataset preparation, the PatchTST baseline, profiling scripts, results helpers,
and the model-side fused-attention integration path.

## What Is Here

### `config.py`
Shared hyperparameters and stable paths used by the baseline pipeline.

### `model/`
Core ETTh1 and PatchTST workflow.

- `data.py`
  - ETTh1 loading, inspection, normalization, windowing, and DataLoaders
- `patchtst.py`
  - PatchTST baseline with a swappable attention block interface
- `train.py`
  - baseline training loop
- `evaluate.py`
  - baseline evaluation script
- `fused_attn_block.py`
  - model-side wrapper for the fused kernel

### `profiling/`
Benchmark scripts.

- `baseline_bench.py`
  - unfused QKV + SDPA benchmark
- `fused_bench.py`
  - canonical fused-kernel benchmark scaffold

### `results/`
Canonical results area for this pipeline.

- CSV merge and figure-generation utilities
- profiling outputs
- figures
- `phase1/` deliverables folder

## Implemented Status

Implemented here:

- ETTh1 preprocessing and baseline training/evaluation flow
- PatchTST baseline model
- baseline profiling script
- fused profiling scaffold in the canonical layout
- model-side fused wrapper scaffold

Partially implemented or still pending:

- full GPU-backed fused-kernel validation from this pipeline
- end-to-end fused PatchTST training with the cleaned layout

## Usage

### Baseline Phase 1 workflow

```bash
cd baseline_pipeline
python model/data.py
python profiling/baseline_bench.py
python model/train.py
python model/evaluate.py
```

### Fused benchmark scaffold

```bash
cd baseline_pipeline
python profiling/fused_bench.py --simulate
```

## Team Work Reflected In This Folder

- Rithwik Amajala
  - ETTh1 pipeline, PatchTST baseline, baseline profiling, training, evaluation
- Bhanuja Karumuru
  - fused benchmark/results scaffolding used for profiling comparisons
- Jnanasree Konda
  - kernel interface assumptions that the fused wrapper and correctness flow rely on
