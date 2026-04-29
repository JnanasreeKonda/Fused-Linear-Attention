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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.attention_utils import (
    convert_state_dict_for_attention,
    infer_attention_from_state_dict,
    normalize_attention_name,
    resolve_attention_block,
)
from model.data import get_dataloaders, FEATURE_COLS, OT_IDX
from model.patchtst import PatchTST


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str = config.CHECKPOINT_PATH,
    out_path: str = config.BASELINE_METRICS_PATH,
    method_name: str = "baseline_unfused",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    attention: str = "standard",
    checkpoint_format: str = "auto",
    no_cuda: bool = False,
) -> dict:
    """
    Load best checkpoint → run on test set → compute MSE & MAE.

    Returns
    -------
    {"method": str, "mse": float, "mae": float}
    """
    attention = normalize_attention_name(attention)
    device = torch.device(
        "cpu" if (no_cuda or not torch.cuda.is_available()) else "cuda"
    )
    print(f"[evaluate] Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python model/train.py  first."
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader, mean, std = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    attn_block_class = resolve_attention_block(attention)
    model = PatchTST(attn_block_class=attn_block_class).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"]
    source_attention = ckpt.get("attention_type") or ckpt.get("config", {}).get("attention_type")
    if checkpoint_format != "auto":
        source_attention = normalize_attention_name(checkpoint_format)
    else:
        source_attention = source_attention or infer_attention_from_state_dict(state_dict)

    if source_attention == "unknown":
        raise RuntimeError(
            "Could not infer checkpoint attention format. "
            "Pass --checkpoint-format standard|fused."
        )

    state_dict = convert_state_dict_for_attention(
        state_dict,
        source_attention=source_attention,
        target_attention=attention,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"[evaluate] Checkpoint  epoch={ckpt['epoch']}  "
        f"val_loss={ckpt['val_loss']:.6f}"
    )
    if source_attention != attention:
        print(
            f"[evaluate] Converted checkpoint attention: "
            f"{source_attention} -> {attention}"
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
    parser.add_argument("--attention", choices=["standard", "fused"], default="standard")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-format", choices=["auto", "standard", "fused"], default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    attention = normalize_attention_name(args.attention)
    checkpoint_path = (
        args.checkpoint
        or (config.FUSED_CHECKPOINT_PATH if attention == "fused" else config.BASELINE_CHECKPOINT_PATH)
    )
    out_path = (
        args.out
        or (config.FUSED_METRICS_PATH if attention == "fused" else config.BASELINE_METRICS_PATH)
    )
    method_name = "fused_kernel" if attention == "fused" else "baseline_unfused"

    evaluate(
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        method_name=method_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attention=attention,
        checkpoint_format=args.checkpoint_format,
        no_cuda=args.no_cuda,
    )


if __name__ == "__main__":
    main()
