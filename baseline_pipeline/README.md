# baseline_pipeline

**Owner: Rithwik Amajala** | NYU Tandon — ECE-GY High Performance ML
**Project: FusedLinearAttention** — A Custom CUDA Kernel Fusing QKV Projection and Attention

---

## What this folder contains

Rithwik's Phase 1 deliverables:

| Milestone | Description | Status |
|-----------|-------------|--------|
| M3 | Baseline NSight profiling — unfused QKV + SDPA microbenchmark on A100 | Complete |
| M5 | ETTh1 data pipeline, PatchTST model, training loop, baseline evaluation | Complete |
| M10 | Fused kernel integration + end-to-end validation (Phase 3) | Pending M8 |
| M11 | All result figures — NSight timeline, HBM bar charts, speedup plots | Pending M9 |

---

## Folder structure

```
baseline_pipeline/
├── config.py                      # Shared hyperparameters — all team use this
├── requirements.txt               # Python dependencies
├── run_phase1.sh                  # One-shot script: M3 + M5 end-to-end
│
├── model/
│   ├── data.py                    # M5: ETTh1 download, EDA, preprocessing, DataLoaders
│   ├── patchtst.py                # M5: PatchTST model with swappable attention interface
│   ├── train.py                   # M5: training loop (Adam, CosineAnnealingLR, early stop)
│   ├── evaluate.py                # M5: test-set MSE/MAE evaluation
│   └── fused_attn_block.py        # M10 placeholder — Phase 3 integration hook
│
├── profiling/
│   └── baseline_bench.py          # M3: 100 warmup + 500 timed iters, CUDA Events, NSight cmds
│
├── results/                       # Auto-generated outputs (gitignored except traces)
│   ├── best_baseline_model.pt     # Best checkpoint — epoch 11, val_loss=0.4779
│   ├── baseline_training_log.csv  # Per-epoch train/val loss, 16 epochs
│   ├── baseline_model_metrics.csv # Test MSE=180.46, MAE=12.65 (de-normalised)
│   ├── baseline_profiling.csv     # Wall-time + peak alloc per seq_len on A100
│   ├── figures/
│   │   └── etth1_ot_column.png    # EDA plot of OT target column
│   └── traces/baseline/
│       ├── baseline.nsys-rep      # NSight Systems trace — open in NSight UI
│       └── baseline.sqlite        # NSight database
│
└── rithwik_report.pdf             # Phase 1 report
```

---

## Quick start (on Greene A100)

```bash
# 1. Clone and enter
git clone https://github.com/JnanasreeKonda/Fused-Linear-Attention.git
cd Fused-Linear-Attention/baseline_pipeline

# 2. Enter CUDA Singularity container
/scratch/work/public/singularity/run-cuda-12.2.bash

# 3. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu122 -q
pip install numpy pandas matplotlib tqdm -q

# 4. Run Phase 1 (M3 + M5)
chmod +x run_phase1.sh
./run_phase1.sh
```

### NSight Systems trace (M3)
```bash
nsys profile \
    --trace=cuda,nvtx \
    --output=results/traces/baseline/baseline \
    python profiling/baseline_bench.py

# Convert to readable summary
nsys stats --report=cuda_gpu_kern_sum \
    results/traces/baseline/baseline.nsys-rep
```

---

## Phase 1 Results (NVIDIA A100-SXM4-40GB)

### Baseline profiling — M3
| seq_len | per_iter (µs) | peak alloc (MB) |
|---------|--------------|-----------------|
| 64 | 302.8 | 12.9 |
| 128 | 305.5 | 13.6 |
| 256 | 301.1 | 15.1 |
| 512 | 270.3 | 18.1 |
| 1024 | 379.6 | 24.1 |

### ETTh1 model baseline — M5
| Metric | Value |
|--------|-------|
| Test MSE (de-normalised) | 180.46 |
| Test MAE (de-normalised) | 12.65 |
| Best val loss | 0.4779 (epoch 11) |
| Total epochs | 16 (early stop, patience=5) |
| Avg epoch time | ~2.17s on A100 |

---

## Key design note — swappable attention interface

`model/patchtst.py` accepts an `attn_block_class` argument. Phase 1 uses
`StandardAttentionBlock` (3 separate `nn.Linear` projections + SDPA — the
two-kernel baseline). Phase 3 (M10) swaps in `FusedLinearAttentionBlock`
with zero model changes:

```python
# Phase 1 baseline
model = PatchTST()

# Phase 3 — drop-in swap (no other changes)
from model.fused_attn_block import FusedLinearAttentionBlock
model = PatchTST(attn_block_class=FusedLinearAttentionBlock)
```

---

## Hardware & software
- **GPU**: NVIDIA A100-SXM4-40GB (NYU Greene cluster)
- **CUDA**: 12.2
- **PyTorch**: 2.x
- **Dataset**: ETTh1 — 17,420 hourly rows, 7 features, 0 missing values
