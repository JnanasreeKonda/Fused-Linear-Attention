"""
model/evaluate.py — Evaluate best baseline checkpoint on ETTh1 test set.
Owner: Rithwik Amajala  |  Milestone: M5  |  Phase 1

Loads the best checkpoint saved during training, runs inference on the
held-out test split, de-normalises predictions and targets, then reports
MSE and MAE.  These numbers are the correctness reference that the fused
kernel (Phase 3, M10) must reproduce within 1%.

Output
------
results/baseline_model_metrics.csv  — columns: method, mse, mae

Usage:
    python model/evaluate.py
    python model/evaluate.py --checkpoint results/best_baseline_model.pt
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.data import get_dataloaders, FEATURE_COLS, OT_IDX
from model.patchtst import PatchTST


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str = config.CHECKPOINT_PATH,
    out_path: str        = "results/baseline_model_metrics.csv",
    method_name: str     = "baseline_unfused",
) -> dict:
    """
    Load best checkpoint → run on test set → compute MSE & MAE.

    Returns
    -------
    {"method": str, "mse": float, "mae": float}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python model/train.py  first."
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader, mean, std = get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PatchTST().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"[evaluate] Checkpoint  epoch={ckpt['epoch']}  "
        f"val_loss={ckpt['val_loss']:.6f}"
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            all_preds.append(model(xb).cpu())
            all_targets.append(yb)

    preds   = torch.cat(all_preds,   dim=0).numpy()   # (N, forecast_len)
    targets = torch.cat(all_targets, dim=0).numpy()   # (N, forecast_len)

    # ── De-normalise using OT scaler ──────────────────────────────────────────
    ot_mean = float(mean[OT_IDX])
    ot_std  = float(std[OT_IDX])
    preds_dn   = preds   * ot_std + ot_mean
    targets_dn = targets * ot_std + ot_mean

    mse = float(np.mean((preds_dn - targets_dn) ** 2))
    mae = float(np.mean(np.abs(preds_dn - targets_dn)))

    print(f"\n[evaluate] ── Baseline Test Metrics ─────────────────────────")
    print(f"  MSE  : {mse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"────────────────────────────────────────────────────────────\n")

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    row = {"method": method_name, "mse": round(mse, 8), "mae": round(mae, 8)}
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"[evaluate] Metrics saved → {out_path}")

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline PatchTST on ETTh1 test set")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    parser.add_argument("--out",        default="results/baseline_model_metrics.csv")
    args = parser.parse_args()

    evaluate(checkpoint_path=args.checkpoint, out_path=args.out)


if __name__ == "__main__":
    main()
