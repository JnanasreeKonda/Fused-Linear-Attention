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
- `end_to_end_validate.py`
  - Phase 3 validation and timing driver
- `attention_utils.py`
  - checkpoint conversion and attention-class helpers

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
- model-side fused wrapper plus Phase 3 validation flow

Partially implemented or still pending:

- full GPU-backed fused-kernel validation from this pipeline
- fused-kernel backward support for true end-to-end training without the
  PyTorch-reference fallback

## Usage

### Baseline Phase 1 workflow

```bash
cd baseline_pipeline
python model/data.py
python profiling/baseline_bench.py
python model/train.py
python model/evaluate.py
```

### Fused profiling workflow

```bash
cd baseline_pipeline
./run_bhanuja.sh --simulate
```

On a compatible CUDA build environment:

```bash
cd baseline_pipeline
./run_bhanuja.sh
```

### Phase 2 / 3 validation helper

```bash
cd baseline_pipeline
./run_phase23.sh
```

### Phase 3 workflow (M10 + M11)

```bash
cd baseline_pipeline
python model/end_to_end_validate.py --no-cuda
python results/merge_comparison.py
python results/generate_figures.py
```

On a CUDA node with a working driver/toolchain, replace `--no-cuda` with
`--train-fused` to retrain the fused attention path and emit the Phase 3
timing/validation artifacts.

## Team Work Reflected In This Folder

- Rithwik Amajala
  - ETTh1 pipeline, PatchTST baseline, baseline profiling, training, evaluation
- Bhanuja Karumuru
  - fused benchmark/results scaffolding used for profiling comparisons
- Jnanasree Konda
  - kernel interface assumptions that the fused wrapper and correctness flow rely on

## Notes

- `run_bhanuja.sh` is the recommended entrypoint for the fused benchmarking
  workflow.
- `baseline_pipeline/results/` is the canonical home for benchmark CSVs,
  comparison tables, generated figures, and preserved deliverables.
- Treat fused results as final performance evidence only when
  `fused_profiling.csv` shows `run_mode=cuda_kernel`.
