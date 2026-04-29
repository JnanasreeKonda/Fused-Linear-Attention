# `baseline_pipeline/results/`

This folder stores canonical outputs for baseline profiling, fused profiling,
comparison tables, generated figures, and official deliverables.

## Main Files

- `baseline_profiling.csv`
  - unfused baseline timing output
- `fused_profiling.csv`
  - fused benchmark timing output
- `occupancy_sweep.csv`
  - tile-size sweep table used for occupancy and tuning analysis
- `comparison_table.csv`
  - merged baseline vs fused profiling summary
- `generate_figures.py`
  - builds final PNG figures
- `merge_comparison.py`
  - merges baseline and fused CSVs

## Subfolders

- `figures/`
  - generated performance plots
- `phase1/`
  - preserved Phase 1 deliverables
- `traces/`
  - NSight traces and profiling artifacts

## Recommended Workflow

From `baseline_pipeline/`:

```bash
./run_bhanuja.sh --simulate
```

On a CUDA-enabled environment:

```bash
./run_bhanuja.sh
```

## Notes

- `fused_profiling.csv` includes a `run_mode` column so simulation output and
  real CUDA-kernel output can be distinguished clearly.
- Only treat CUDA-backed profiling outputs as official performance results.
