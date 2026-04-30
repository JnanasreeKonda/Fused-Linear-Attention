"""
results/merge_comparison.py — Merge baseline and fused profiling CSVs safely.

This script refuses to report latency speedups when the two inputs are not
directly comparable, for example:
- baseline measured on CUDA but fused run in CPU simulation mode
- fused run using the PyTorch simulation path instead of the compiled kernel
- mismatched GPU targets
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, Tuple


def load_csv_keyed(path: str, key: str = "seq_len") -> Dict:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row[key])] = row
    return rows


def safe_float(row: dict, col: str):
    value = row.get(col, "")
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def infer_execution_mode(row: dict, method: str) -> str:
    mode = row.get("execution_mode")
    if mode:
        return mode

    device = (row.get("device") or "").strip().lower()
    if device == "cuda":
        return "cuda_measured"
    return "cpu_fallback"


def infer_kernel_backend(row: dict, method: str) -> str:
    backend = row.get("kernel_backend")
    if backend:
        return backend

    if method == "baseline_unfused":
        return "pytorch_unfused"
    if (row.get("device") or "").strip().lower() != "cuda":
        return "simulate_reference"
    return "compiled_cuda_kernel"


def estimate_hbm_bytes(method: str, seq_len: int, embed_dim: int, n_heads: int, batch_size: int):
    d_head = embed_dim // n_heads
    fp32_bytes = 4

    if method == "baseline_unfused":
        read_bytes = (
            batch_size * seq_len * embed_dim
            + 3 * embed_dim * n_heads * d_head
            + 3 * batch_size * n_heads * seq_len * d_head
        ) * fp32_bytes
        write_bytes = (4 * batch_size * n_heads * seq_len * d_head) * fp32_bytes
    else:
        read_bytes = (
            batch_size * seq_len * embed_dim
            + 3 * embed_dim * n_heads * d_head
        ) * fp32_bytes
        write_bytes = (batch_size * n_heads * seq_len * d_head) * fp32_bytes

    return read_bytes, write_bytes


def comparison_status(baseline_row: dict, fused_row: dict) -> Tuple[str, str]:
    b_exec = infer_execution_mode(baseline_row, "baseline_unfused")
    f_exec = infer_execution_mode(fused_row, "fused_kernel")
    b_backend = infer_kernel_backend(baseline_row, "baseline_unfused")
    f_backend = infer_kernel_backend(fused_row, "fused_kernel")
    b_gpu = baseline_row.get("gpu_name", "")
    f_gpu = fused_row.get("gpu_name", "")

    reasons = []
    if b_exec != "cuda_measured":
        reasons.append(f"baseline run mode is {b_exec}")
    if f_exec != "cuda_measured":
        reasons.append(f"fused run mode is {f_exec}")
    if f_backend != "compiled_cuda_kernel":
        reasons.append(f"fused backend is {f_backend}")
    if b_exec == "cuda_measured" and f_exec == "cuda_measured" and b_gpu and f_gpu and b_gpu != f_gpu:
        reasons.append(f"GPU mismatch ({b_gpu} vs {f_gpu})")

    if reasons:
        return "incompatible", "; ".join(reasons)
    return "comparable", "same-device CUDA measurements"


def main():
    results_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(results_dir, "baseline_profiling.csv")
    fused_path = os.path.join(results_dir, "fused_profiling.csv")
    out_path = os.path.join(results_dir, "comparison_table.csv")

    if not os.path.exists(base_path):
        sys.exit(f"ERROR: {base_path} not found.")
    if not os.path.exists(fused_path):
        sys.exit(f"ERROR: {fused_path} not found.")

    baseline = load_csv_keyed(base_path)
    fused = load_csv_keyed(fused_path)
    seq_lens = sorted(set(baseline) & set(fused))
    if not seq_lens:
        sys.exit("ERROR: no overlapping seq_len rows between profiling CSVs.")

    output_rows = []
    for seq_len in seq_lens:
        b = baseline[seq_len]
        f = fused[seq_len]

        status, notes = comparison_status(b, f)

        b_embed = int(b.get("embed_dim", 512))
        b_heads = int(b.get("n_heads", 8))
        b_batch = int(b.get("batch_size", 1))
        f_embed = int(f.get("embed_dim", 512))
        f_heads = int(f.get("n_heads", 8))
        f_batch = int(f.get("batch_size", 1))

        b_read, b_write = estimate_hbm_bytes("baseline_unfused", seq_len, b_embed, b_heads, b_batch)
        f_read, f_write = estimate_hbm_bytes("fused_kernel", seq_len, f_embed, f_heads, f_batch)

        row_b_read = safe_float(b, "HBM_read_bytes_est")
        row_b_write = safe_float(b, "HBM_write_bytes_est")
        row_f_read = safe_float(f, "HBM_read_bytes_est")
        row_f_write = safe_float(f, "HBM_write_bytes_est")
        if row_b_read is not None:
            b_read = row_b_read
        if row_b_write is not None:
            b_write = row_b_write
        if row_f_read is not None:
            f_read = row_f_read
        if row_f_write is not None:
            f_write = row_f_write

        b_time = safe_float(b, "per_iter_us")
        f_time = safe_float(f, "per_iter_us")
        speedup = (
            round(b_time / f_time, 4)
            if status == "comparable" and b_time and f_time and f_time > 0
            else ""
        )
        hbm_reduction = (
            round((1.0 - f_read / b_read) * 100, 2)
            if (b_read and f_read and b_read > 0)
            else ""
        )

        for method, row, is_fused, read_bytes, write_bytes in [
            ("baseline_unfused", b, False, b_read, b_write),
            ("fused_kernel", f, True, f_read, f_write),
        ]:
            output_rows.append(
                {
                    "method": method,
                    "run_mode": row.get("run_mode", "baseline" if not is_fused else "unknown"),
                    "seq_len": seq_len,
                    "embed_dim": row.get("embed_dim", 512),
                    "n_heads": row.get("n_heads", 8),
                    "batch_size": row.get("batch_size", 1),
                    "warmup_iters": row.get("warmup_iters"),
                    "timed_iters": row.get("timed_iters"),
                    "total_elapsed_ms": row.get("total_elapsed_ms"),
                    "per_iter_us": row.get("per_iter_us"),
                    "peak_alloc_mb": row.get("peak_alloc_mb"),
                    "kernel_count": row.get("kernel_count", "1" if is_fused else "2"),
                    "HBM_read_bytes_est": int(read_bytes) if read_bytes is not None else "",
                    "HBM_write_bytes_est": int(write_bytes) if write_bytes is not None else "",
                    "device": row.get("device"),
                    "gpu_name": row.get("gpu_name"),
                    "execution_mode": infer_execution_mode(row, method),
                    "kernel_backend": infer_kernel_backend(row, method),
                    "comparison_status": status,
                    "comparison_notes": notes,
                    "speedup_vs_baseline": speedup if is_fused else "",
                    "HBM_read_reduction_pct": hbm_reduction if is_fused else "",
                }
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved: {out_path}")
    print()
    print("Summary:")
    for row in output_rows:
        if row["method"] != "fused_kernel":
            continue
        print(
            f"  seq_len={row['seq_len']:>4}  "
            f"mode={row['run_mode']:<11}  "
            f"speedup={row['speedup_vs_baseline'] or 'N/A':>6}  "
            f"hbm_reduction={row['HBM_read_reduction_pct'] or 'N/A'}%"
        )


if __name__ == "__main__":
    main()
