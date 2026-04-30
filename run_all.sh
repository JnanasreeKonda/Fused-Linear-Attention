#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

WITH_GOLDEN=0
for arg in "$@"; do
  if [[ "$arg" == "--with-golden" ]]; then
    WITH_GOLDEN=1
  fi
done

echo "========== BASELINE PIPELINE (PHASE 1) =========="
bash baseline_pipeline/run_phase1.sh

echo "========== FUSED PROFILING (PHASE 3 / BHANUJA) =========="
if python3 - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  (
    cd baseline_pipeline
    python3 profiling/fused_bench.py
  )
else
  echo "CUDA unavailable here; running fused benchmark in simulation mode."
  (
    cd baseline_pipeline
    python3 profiling/fused_bench.py --simulate
  )
fi

echo "========== FUSED PIPELINE (PHASE 3) =========="
bash baseline_pipeline/run_phase3.sh --train-fused

echo "========== KERNEL CORRECTNESS TESTS =========="
python3 tests/test_correctness.py --out baseline_pipeline/results/correctness_results.csv

echo "========== CPU NUMPY REFERENCE TESTS =========="
(
  cd CPU_Reference_in_NumPy/tests
  python3 test_reference_vs_pytorch.py
)

if [[ "$WITH_GOLDEN" == "1" ]]; then
  echo "========== GOLDEN OUTPUTS (OPTIONAL) =========="
  (
    cd CPU_Reference_in_NumPy/tests
    python3 generate_golden_outputs.py
  )
else
  echo "========== GOLDEN OUTPUTS (OPTIONAL) =========="
  echo "Skipping golden output regeneration. Pass --with-golden to enable it."
fi

echo "========== ALL DONE =========="
