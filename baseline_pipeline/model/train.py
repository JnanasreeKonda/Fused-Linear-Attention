"""
model/train.py — Training loop for PatchTST baseline on ETTh1.
Owner: Rithwik Amajala  |  Milestone: M5  |  Phase 1

Config (all values from config.py):
  Optimiser  : Adam  lr=1e-4, weight_decay=1e-4
  LR schedule: CosineAnnealingLR  (T_max=EPOCHS, eta_min=lr*0.01)
  Loss       : MSELoss
  Gradient clipping: max_norm=1.0
  Epochs     : 20, early stopping patience=5 (monitored on val MSE)
  Checkpoint : results/best_baseline_model.pt  (best val_loss)
  Log CSV    : results/baseline_training_log.csv

Usage:
    python model/train.py
    python model/train.py --epochs 10 --no-cuda      # quick test
"""

import argparse
import csv
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.attention_utils import normalize_attention_name, resolve_attention_block
from model.data import get_dataloaders
from model.patchtst import PatchTST


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# Core training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    patience: int = config.PATIENCE,
    checkpoint_path: str = config.CHECKPOINT_PATH,
    log_path: str = config.BASELINE_TRAIN_LOG_PATH,
    gradient_log_path: str | None = None,
    attention_name: str = "standard",
) -> float:
    """
    Train `model` on `train_loader`, validate on `val_loader`.

    Saves the best checkpoint to `checkpoint_path` and per-epoch
    metrics to `log_path`.

    Returns
    -------
    best_val_loss : float
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    criterion = nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss     = float("inf")
    patience_counter  = 0
    log_rows: list = []
    grad_rows: list = []
    attention_name = normalize_attention_name(attention_name)

    print(f"\n[train] Starting {attention_name} training for {epochs} epochs (patience={patience})")
    print(f"[train] Checkpoint → {checkpoint_path}")
    print(f"[train] Log        → {log_path}\n")
    if gradient_log_path:
        print(f"[train] Grad log   → {gradient_log_path}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_loss = 0.0
        grad_norms = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            grad_sq_sum = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad_norm = float(param.grad.detach().float().norm(2).item())
                if not np.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm detected during epoch {epoch} "
                        f"({attention_name} attention)."
                    )
                grad_sq_sum += grad_norm ** 2
            grad_norms.append(grad_sq_sum ** 0.5)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(train_loader.dataset)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        elapsed   = time.time() - t0
        lr_now    = scheduler.get_last_lr()[0]
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        max_grad_norm = float(np.max(grad_norms)) if grad_norms else 0.0

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"train={train_loss:.6f}  val={val_loss:.6f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s | "
            f"grad(avg/max)={avg_grad_norm:.3f}/{max_grad_norm:.3f}"
        )

        log_rows.append(
            dict(
                epoch=epoch,
                train_loss=round(train_loss, 8),
                val_loss=round(val_loss, 8),
                lr=lr_now,
                elapsed_s=round(elapsed, 2),
                avg_grad_norm=round(avg_grad_norm, 8),
                max_grad_norm=round(max_grad_norm, 8),
            )
        )
        grad_rows.append(
            dict(
                epoch=epoch,
                attention=attention_name,
                avg_grad_norm=round(avg_grad_norm, 8),
                max_grad_norm=round(max_grad_norm, 8),
                train_loss=round(train_loss, 8),
                val_loss=round(val_loss, 8),
            )
        )

        # ── Early stopping + checkpoint ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "train_loss":  train_loss,
                    "config": {
                        "d_model":      config.D_MODEL,
                        "n_heads":      config.N_HEADS,
                        "n_layers":     config.N_LAYERS,
                        "patch_len":    config.PATCH_LEN,
                        "stride":       config.STRIDE,
                        "input_len":    config.INPUT_LEN,
                        "forecast_len": config.FORECAST_LEN,
                        "attention_type": attention_name,
                    },
                    "attention_type": attention_name,
                },
                checkpoint_path,
            )
            print(f"    ✓ checkpoint saved (best val={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[train] Early stopping at epoch {epoch} (patience={patience}).")
                break

    # ── Write training log ────────────────────────────────────────────────────
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    if gradient_log_path and grad_rows:
        with open(gradient_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=grad_rows[0].keys())
            writer.writeheader()
            writer.writerows(grad_rows)

    print(f"\n[train] Done. Best val MSE = {best_val_loss:.6f}")
    print(f"[train] Log saved → {log_path}")
    if gradient_log_path:
        print(f"[train] Grad log saved → {gradient_log_path}")
    return best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PatchTST baseline on ETTh1")
    parser.add_argument("--attention", choices=["standard", "fused"], default="standard")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--grad-log", default=None)
    args = parser.parse_args()

    set_seed()
    attention_name = normalize_attention_name(args.attention)

    device = torch.device(
        "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    )
    print(f"[train] Device : {device}")
    if device.type == "cuda":
        print(f"[train] GPU    : {torch.cuda.get_device_name(device)}")

    train_loader, val_loader, _, _, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    attn_block_class = resolve_attention_block(attention_name)
    effective_dropout = args.dropout
    if attention_name == "fused" and device.type == "cuda" and effective_dropout > 0:
        print(
            "[train] Fused CUDA attention does not implement attention-weight "
            "dropout in the custom kernel yet; forcing dropout=0.0 for the "
            "fused training run."
        )
        effective_dropout = 0.0

    model = PatchTST(
        attn_block_class=attn_block_class,
        dropout=effective_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters : {n_params:,}")

    checkpoint_path = (
        args.checkpoint
        or (config.FUSED_CHECKPOINT_PATH if attention_name == "fused" else config.BASELINE_CHECKPOINT_PATH)
    )
    log_path = (
        args.log
        or (config.FUSED_TRAIN_LOG_PATH if attention_name == "fused" else config.BASELINE_TRAIN_LOG_PATH)
    )
    grad_log_path = args.grad_log
    if grad_log_path is None and attention_name == "fused":
        grad_log_path = config.FUSED_GRAD_LOG_PATH

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        gradient_log_path=grad_log_path,
        attention_name=attention_name,
    )


if __name__ == "__main__":
    main()
