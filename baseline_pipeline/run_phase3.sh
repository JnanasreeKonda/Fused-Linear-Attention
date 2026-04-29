#!/usr/bin/env bash
# run_phase3.sh — Run Rithwik's Phase 3 tasks (M10 + M11).
#
# Greene / CUDA workflow:
#   chmod +x run_phase3.sh && ./run_phase3.sh
#
# CPU/reference-path verification:
#   chmod +x run_phase3.sh && ./run_phase3.sh --no-cuda

set -e
cd "$(dirname "$0")"

NO_CUDA=""
TRAIN_FUSED=""
for arg in "$@"; do
  if [[ "$arg" == "--no-cuda" ]]; then
    NO_CUDA="--no-cuda"
  fi
  if [[ "$arg" == "--train-fused" ]]; then
    TRAIN_FUSED="--train-fused"
  fi
done

echo "═══════════════════════════════════════════════════════════════"
echo " FusedLinearAttention — Phase 3 (Rithwik)"
echo "═══════════════════════════════════════════════════════════════"

echo -e "\n[1/4] M10 — End-to-end validation and timing …"
python3 model/end_to_end_validate.py ${NO_CUDA} ${TRAIN_FUSED}

echo -e "\n[2/4] Merge baseline/fused profiling tables …"
if [[ -f results/baseline_profiling.csv && -f results/fused_profiling.csv ]]; then
  python3 results/merge_comparison.py
else
  echo "  Skipping merge: profiling CSVs not both present."
fi

echo -e "\n[3/4] M11 — Generate Phase 3 figures …"
if [[ -f results/comparison_table.csv ]]; then
  python3 results/generate_figures.py
else
  echo "  Skipping figures: results/comparison_table.csv not found."
fi

echo -e "\n[4/4] Outputs"
echo "  results/endtoend_timing.csv"
echo "  results/validation_table.csv"
echo "  results/fused_model_metrics.csv"
echo "  results/figures/"
echo "═══════════════════════════════════════════════════════════════"
