#!/usr/bin/env bash
# run_phase23.sh — Run the Phase 2 and Phase 3 validation flow.
#
# Usage:
#   cd baseline_pipeline
#   chmod +x run_phase23.sh
#   ./run_phase23.sh

set -e
cd "$(dirname "$0")"

if [[ -x "../.venv/bin/python" ]]; then
  PYTHON_BIN="../.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "═══════════════════════════════════════════════════════════════"
echo " FusedLinearAttention — Phase 2 / Phase 3"
echo "═══════════════════════════════════════════════════════════════"

echo -e "\n[1/4] NumPy oracle smoke-check …"
"$PYTHON_BIN" ../tests/reference.py

echo -e "\n[2/4] Correctness suite (PyTorch simulation, quick) …"
"$PYTHON_BIN" ../tests/test_correctness.py --simulate --quick --use-root-oracle

echo -e "\n[3/4] Fused benchmark scaffold (PyTorch simulation) …"
"$PYTHON_BIN" profiling/fused_bench.py --simulate --timed 20 --warmup 5

echo -e "\n[4/4] Notes …"
echo "For the compiled CUDA kernel path on a CUDA-enabled PyTorch environment:"
echo "  python3 ../tests/test_correctness.py --quick"
echo "  python3 profiling/fused_bench.py"
echo ""
echo "If you want to instantiate the fused model-side block:"
echo "  from model.patchtst import PatchTST"
echo "  from model.fused_attn_block import FusedLinearAttentionBlock"
echo "  model = PatchTST(attn_block_class=FusedLinearAttentionBlock)"
