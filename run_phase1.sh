#!/usr/bin/env bash
# run_phase1.sh — Run all Phase 1 tasks for Rithwik (M3 + M5) in order.
# Usage on Greene A100 node:
#   chmod +x run_phase1.sh && ./run_phase1.sh

set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════"
echo " FusedLinearAttention — Phase 1 (Rithwik)"
echo "═══════════════════════════════════════════════════════════════"

# ── Install dependencies ─────────────────────────────────────────────────────
echo -e "\n[0/4] Installing requirements …"
pip install -q -r requirements.txt

# ── M5a: Data pipeline smoke-test ────────────────────────────────────────────
echo -e "\n[1/4] M5 — ETTh1 pipeline smoke-test …"
python3 model/data.py

# ── M3: Baseline NSight microbenchmark (wall-time only) ──────────────────────
echo -e "\n[2/4] M3 — Baseline profiling (wall-time) …"
python3 profiling/baseline_bench.py

# ── M5b: Train PatchTST baseline ─────────────────────────────────────────────
echo -e "\n[3/4] M5 — Training PatchTST baseline …"
python3 model/train.py

# ── M5c: Evaluate baseline on test set ───────────────────────────────────────
echo -e "\n[4/4] M5 — Evaluating on test set …"
python3 model/evaluate.py

echo -e "\n═══════════════════════════════════════════════════════════════"
echo " Phase 1 complete.  Key outputs:"
echo "   results/baseline_profiling.csv"
echo "   results/baseline_training_log.csv"
echo "   results/baseline_model_metrics.csv"
echo "   results/best_baseline_model.pt"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " NSight Systems trace (run on Greene A100):"
echo "   nsys profile --trace=cuda,nvtx \\"
echo "        --output=results/traces/baseline/baseline \\"
echo "        python profiling/baseline_bench.py"
