"""
model/end_to_end_validate.py — Phase 3 (M10) end-to-end validation driver.

Outputs
-------
- results/endtoend_timing.csv
- results/validation_table.csv
- results/fused_model_metrics.csv
- results/fused_training_log.csv           (when --train-fused is used)
- results/fused_gradient_norms.csv         (when --train-fused is used)

Usage
-----
CPU/reference-path verification:
    python model/end_to_end_validate.py --no-cuda

Full fused retraining on a CUDA node with the compiled kernel available:
    python model/end_to_end_validate.py --train-fused
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.attention_utils import (
    convert_state_dict_for_attention,
    infer_attention_from_state_dict,
    normalize_attention_name,
    resolve_attention_block,
)
from model.data import get_dataloaders
from model.evaluate import evaluate
from model.patchtst import PatchTST
from model.train import set_seed, train


def summarize_training_log(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    elapsed = [float(row["elapsed_s"]) for row in rows if row.get("elapsed_s")]
    if not elapsed:
        return {}

    return {
        "epochs_logged": len(elapsed),
        "epoch_time_mean_s": round(sum(elapsed) / len(elapsed), 4),
        "epoch_time_min_s": round(min(elapsed), 4),
        "epoch_time_max_s": round(max(elapsed), 4),
    }


def load_model_for_attention(
    attention: str,
    checkpoint_path: str,
    checkpoint_format: str,
    device: torch.device,
):
    attention = normalize_attention_name(attention)
    attn_block_class = resolve_attention_block(attention)
    model = PatchTST(attn_block_class=attn_block_class).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"]
    source_attention = ckpt.get("attention_type") or ckpt.get("config", {}).get(
        "attention_type"
    )
    if checkpoint_format != "auto":
        source_attention = normalize_attention_name(checkpoint_format)
    else:
        source_attention = source_attention or infer_attention_from_state_dict(state_dict)

    if source_attention == "unknown":
        raise RuntimeError(
            f"Could not infer attention format for checkpoint: {checkpoint_path}"
        )

    state_dict = convert_state_dict_for_attention(
        state_dict,
        source_attention=source_attention,
        target_attention=attention,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, source_attention


def benchmark_forward(
    model: torch.nn.Module,
    xb: torch.Tensor,
    device: torch.device,
    warmup: int,
    timed: int,
) -> dict:
    xb = xb.to(device, non_blocking=True)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(timed):
                _ = model(xb)
        end.record()
        torch.cuda.synchronize(device)
        total_ms = float(start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(timed):
                _ = model(xb)
        total_ms = (time.perf_counter() - t0) * 1e3

    return {
        "forward_total_ms": round(total_ms, 6),
        "forward_per_iter_ms": round(total_ms / timed, 6),
        "batch_size": int(xb.size(0)),
        "input_len": int(xb.size(1)),
        "n_features": int(xb.size(2)),
        "warmup_iters": int(warmup),
        "timed_iters": int(timed),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run M10 end-to-end validation")
    parser.add_argument("--train-fused", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--baseline-checkpoint", default=config.BASELINE_CHECKPOINT_PATH)
    parser.add_argument("--baseline-checkpoint-format", choices=["auto", "standard", "fused"], default="auto")
    parser.add_argument("--baseline-log", default=config.BASELINE_TRAIN_LOG_PATH)
    parser.add_argument("--fused-checkpoint", default=config.FUSED_CHECKPOINT_PATH)
    parser.add_argument("--fused-checkpoint-format", choices=["auto", "standard", "fused"], default="auto")
    parser.add_argument("--fused-log", default=config.FUSED_TRAIN_LOG_PATH)
    parser.add_argument("--fused-grad-log", default=config.FUSED_GRAD_LOG_PATH)
    parser.add_argument("--baseline-metrics-out", default=config.BASELINE_METRICS_PATH)
    parser.add_argument("--fused-metrics-out", default=config.FUSED_METRICS_PATH)
    parser.add_argument("--validation-out", default=config.VALIDATION_TABLE_PATH)
    parser.add_argument("--timing-out", default=config.ENDTOEND_TIMING_PATH)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--forward-warmup", type=int, default=config.FORWARD_WARMUP_ITERS)
    parser.add_argument("--forward-iters", type=int, default=config.FORWARD_TIMED_ITERS)
    args = parser.parse_args()

    device = torch.device(
        "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    )
    print(f"[m10] Device: {device}")

    if device.type == "cpu" and args.num_workers > 0:
        print(
            "[m10] CPU mode detected; forcing num_workers=0 to avoid "
            "shared-memory worker startup issues in restricted environments."
        )
        args.num_workers = 0

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.train_fused:
        set_seed()
        fused_model = PatchTST(
            attn_block_class=resolve_attention_block("fused")
        ).to(device)
        print("[m10] Training fused-attention PatchTST from scratch...")
        train(
            fused_model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            checkpoint_path=args.fused_checkpoint,
            log_path=args.fused_log,
            gradient_log_path=args.fused_grad_log,
            attention_name="fused",
        )

    baseline_metrics = evaluate(
        checkpoint_path=args.baseline_checkpoint,
        out_path=args.baseline_metrics_out,
        method_name="baseline_unfused",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attention="standard",
        checkpoint_format=args.baseline_checkpoint_format,
        no_cuda=args.no_cuda,
    )

    fused_checkpoint_path = args.fused_checkpoint
    fused_checkpoint_format = args.fused_checkpoint_format
    fused_note = ""
    if not os.path.exists(fused_checkpoint_path):
        fused_checkpoint_path = args.baseline_checkpoint
        fused_checkpoint_format = "standard"
        fused_note = "converted_from_baseline_checkpoint"
        print(
            "[m10] Fused checkpoint not found. Using the baseline checkpoint "
            "converted into the fused attention parameter format for validation."
        )

    fused_metrics = evaluate(
        checkpoint_path=fused_checkpoint_path,
        out_path=args.fused_metrics_out,
        method_name="fused_kernel",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attention="fused",
        checkpoint_format=fused_checkpoint_format,
        no_cuda=args.no_cuda,
    )

    xb, _ = next(iter(test_loader))
    baseline_model, baseline_source = load_model_for_attention(
        attention="standard",
        checkpoint_path=args.baseline_checkpoint,
        checkpoint_format=args.baseline_checkpoint_format,
        device=device,
    )
    fused_model, fused_source = load_model_for_attention(
        attention="fused",
        checkpoint_path=fused_checkpoint_path,
        checkpoint_format=fused_checkpoint_format,
        device=device,
    )

    baseline_forward = benchmark_forward(
        baseline_model, xb, device, args.forward_warmup, args.forward_iters
    )
    fused_forward = benchmark_forward(
        fused_model, xb, device, args.forward_warmup, args.forward_iters
    )

    baseline_log_stats = summarize_training_log(args.baseline_log)
    fused_log_stats = summarize_training_log(args.fused_log)

    baseline_forward_ms = baseline_forward["forward_per_iter_ms"]
    fused_forward_ms = fused_forward["forward_per_iter_ms"]
    forward_speedup = (
        round(baseline_forward_ms / fused_forward_ms, 6)
        if fused_forward_ms > 0
        else ""
    )

    baseline_epoch_mean = baseline_log_stats.get("epoch_time_mean_s", "")
    fused_epoch_mean = fused_log_stats.get("epoch_time_mean_s", "")
    epoch_speedup = (
        round(float(baseline_epoch_mean) / float(fused_epoch_mean), 6)
        if baseline_epoch_mean != "" and fused_epoch_mean not in {"", 0, 0.0}
        else ""
    )

    timing_rows = [
        {
            "method": "baseline_unfused",
            "device": str(device),
            "checkpoint_path": args.baseline_checkpoint,
            "checkpoint_source_attention": baseline_source,
            "mean_epoch_time_s": baseline_epoch_mean,
            "min_epoch_time_s": baseline_log_stats.get("epoch_time_min_s", ""),
            "max_epoch_time_s": baseline_log_stats.get("epoch_time_max_s", ""),
            "forward_per_iter_ms": baseline_forward_ms,
            "forward_total_ms": baseline_forward["forward_total_ms"],
            "batch_size": baseline_forward["batch_size"],
            "input_len": baseline_forward["input_len"],
            "n_features": baseline_forward["n_features"],
            "warmup_iters": baseline_forward["warmup_iters"],
            "timed_iters": baseline_forward["timed_iters"],
            "speedup_vs_baseline": "",
            "notes": "reference_baseline",
        },
        {
            "method": "fused_kernel",
            "device": str(device),
            "checkpoint_path": fused_checkpoint_path,
            "checkpoint_source_attention": fused_source,
            "mean_epoch_time_s": fused_epoch_mean,
            "min_epoch_time_s": fused_log_stats.get("epoch_time_min_s", ""),
            "max_epoch_time_s": fused_log_stats.get("epoch_time_max_s", ""),
            "forward_per_iter_ms": fused_forward_ms,
            "forward_total_ms": fused_forward["forward_total_ms"],
            "batch_size": fused_forward["batch_size"],
            "input_len": fused_forward["input_len"],
            "n_features": fused_forward["n_features"],
            "warmup_iters": fused_forward["warmup_iters"],
            "timed_iters": fused_forward["timed_iters"],
            "speedup_vs_baseline": forward_speedup,
            "notes": fused_note or ("cpu_reference_fallback" if device.type != "cuda" else "phase3_fused_path"),
        },
    ]
    if epoch_speedup != "":
        timing_rows[1]["epoch_speedup_vs_baseline"] = epoch_speedup
        timing_rows[0]["epoch_speedup_vs_baseline"] = ""
    else:
        timing_rows[0]["epoch_speedup_vs_baseline"] = ""
        timing_rows[1]["epoch_speedup_vs_baseline"] = ""

    mse_delta_pct = (
        ((fused_metrics["mse"] - baseline_metrics["mse"]) / baseline_metrics["mse"]) * 100.0
        if baseline_metrics["mse"] != 0
        else 0.0
    )
    mae_delta_pct = (
        ((fused_metrics["mae"] - baseline_metrics["mae"]) / baseline_metrics["mae"]) * 100.0
        if baseline_metrics["mae"] != 0
        else 0.0
    )
    within_1pct = abs(mse_delta_pct) <= 1.0 and abs(mae_delta_pct) <= 1.0

    validation_rows = [
        {
            "method": "baseline_unfused",
            "mse": round(baseline_metrics["mse"], 8),
            "mae": round(baseline_metrics["mae"], 8),
            "mse_delta_pct_vs_baseline": 0.0,
            "mae_delta_pct_vs_baseline": 0.0,
            "within_1pct": "reference",
            "checkpoint_path": args.baseline_checkpoint,
        },
        {
            "method": "fused_kernel",
            "mse": round(fused_metrics["mse"], 8),
            "mae": round(fused_metrics["mae"], 8),
            "mse_delta_pct_vs_baseline": round(mse_delta_pct, 8),
            "mae_delta_pct_vs_baseline": round(mae_delta_pct, 8),
            "within_1pct": "YES" if within_1pct else "NO",
            "checkpoint_path": fused_checkpoint_path,
        },
    ]

    write_csv(args.timing_out, timing_rows)
    write_csv(args.validation_out, validation_rows)

    print(f"[m10] Saved timing CSV      -> {args.timing_out}")
    print(f"[m10] Saved validation CSV  -> {args.validation_out}")
    print(
        f"[m10] Fused vs baseline Δ: "
        f"MSE={mse_delta_pct:.4f}%  MAE={mae_delta_pct:.4f}%  "
        f"within_1pct={within_1pct}"
    )


if __name__ == "__main__":
    main()
