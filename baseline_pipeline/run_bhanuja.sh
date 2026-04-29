#!/usr/bin/env bash
# run_bhanuja.sh — Bhanuja's profiling and figure pipeline.
#
# Usage:
#   cd baseline_pipeline
#   chmod +x run_bhanuja.sh
#   ./run_bhanuja.sh --simulate
#   ./run_bhanuja.sh

set -e
cd "$(dirname "$0")"

if [[ -x "../.venv/bin/python" ]]; then
  PYTHON_BIN="../.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="${PWD}/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

MODE="cuda"
if [[ "${1:-}" == "--simulate" ]]; then
  MODE="simulate"
fi

echo "==============================================================="
echo " FusedLinearAttention — Bhanuja profiling pipeline"
echo "==============================================================="

if [[ "$MODE" == "simulate" ]]; then
  echo
  echo "[1/4] Fused benchmark (simulation) ..."
  "$PYTHON_BIN" profiling/fused_bench.py --simulate --warmup 5 --timed 20
else
  echo
  echo "[1/4] Fused benchmark (CUDA kernel) ..."
  "$PYTHON_BIN" profiling/fused_bench.py
fi

echo
echo "[2/4] Merge baseline and fused profiling CSVs ..."
"$PYTHON_BIN" results/merge_comparison.py

echo
echo "[3/4] Generate figures ..."
"$PYTHON_BIN" results/generate_figures.py

echo
echo "[4/4] Outputs"
echo "  results/fused_profiling.csv"
echo "  results/occupancy_sweep.csv"
echo "  results/comparison_table.csv"
echo "  results/figures/hbm_bandwidth.png"
echo "  results/figures/speedup.png"
echo "  results/figures/occupancy_vs_tile.png"
echo "  results/figures/kernel_count.png"
echo
echo "If you are on GPU and need occupancy data, fill these next:"
echo "  results/occupancy_sweep.csv -> wall_time_ms, SM_occupancy_pct"
