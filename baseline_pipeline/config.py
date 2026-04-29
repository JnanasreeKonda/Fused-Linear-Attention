"""
config.py — Shared hyperparameters for FusedLinearAttention.
All team members must use these values unchanged.

FusedLinearAttention — A Custom CUDA Kernel Fusing QKV Projection and
Attention for Transformer Inference.
Team: Bhanuja · Jnanasree · Rithwik Amajala
NYU Tandon — ECE-GY High Performance ML
"""

import os

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASELINE_DIR)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
DATA_PATH = os.path.join(REPO_ROOT, "data", "ETTh1.csv")

INPUT_LEN     = 96    # lookback window (hours)
FORECAST_LEN  = 96    # forecast horizon (hours)

# Standard ETTh1 splits (hours)
TRAIN_SIZE = 8760     # ≈ 12 months
VAL_SIZE   = 2880     # ≈ 4 months
TEST_SIZE  = 2880     # ≈ 4 months

TARGET_COL    = "OT"  # target column for forecasting

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 2
PATCH_LEN = 16
STRIDE    = 8
D_FF      = 256
DROPOUT   = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
EPOCHS         = 20
PATIENCE       = 5       # early-stopping patience (val-loss)
SEED           = 42
NUM_WORKERS    = 4

CHECKPOINT_DIR  = os.path.join(BASELINE_DIR, "results")
RESULTS_DIR = CHECKPOINT_DIR
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")

BASELINE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_baseline_model.pt")
FUSED_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_fused_model.pt")
CHECKPOINT_PATH = BASELINE_CHECKPOINT_PATH

BASELINE_TRAIN_LOG_PATH = os.path.join(RESULTS_DIR, "baseline_training_log.csv")
FUSED_TRAIN_LOG_PATH = os.path.join(RESULTS_DIR, "fused_training_log.csv")
FUSED_GRAD_LOG_PATH = os.path.join(RESULTS_DIR, "fused_gradient_norms.csv")

BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, "baseline_model_metrics.csv")
FUSED_METRICS_PATH = os.path.join(RESULTS_DIR, "fused_model_metrics.csv")
VALIDATION_TABLE_PATH = os.path.join(RESULTS_DIR, "validation_table.csv")
ENDTOEND_TIMING_PATH = os.path.join(RESULTS_DIR, "endtoend_timing.csv")

# ── Profiling ─────────────────────────────────────────────────────────────────
WARMUP_ITERS    = 100
TIMED_ITERS     = 500
FORWARD_WARMUP_ITERS = 25
FORWARD_TIMED_ITERS = 100
SEQ_LENGTHS     = [64, 128, 256, 512, 1024]
D_HEAD          = 64
N_HEADS_BENCH   = 8
BATCH_BENCH     = 1
EMBED_DIM_BENCH = N_HEADS_BENCH * D_HEAD   # 512
