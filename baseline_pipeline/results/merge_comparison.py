"""
results/merge_comparison.py — Merge baseline and fused profiling CSVs.
"""

from __future__ import annotations

import csv
import os
import sys


def load_csv_keyed(path: str, key: str = "seq_len") -> dict:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row[key])] = row
    return rows


def safe_float(row: dict, col: str):
    v = row.get(col, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


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

    output_rows = []
    for seq_len in seq_lens:
        b = baseline[seq_len]
        f = fused[seq_len]

        b_time = safe_float(b, "per_iter_us")
        f_time = safe_float(f, "per_iter_us")
        speedup = round(b_time / f_time, 4) if (b_time and f_time and f_time > 0) else ""

        b_hbm = safe_float(b, "HBM_read_bytes_est")
        f_hbm = safe_float(f, "HBM_read_bytes_est")
        hbm_reduction = (
            round((1.0 - f_hbm / b_hbm) * 100, 2)
            if (b_hbm and f_hbm and b_hbm > 0)
            else ""
        )

        for method, row, is_fused in [("baseline_unfused", b, False), ("fused_kernel", f, True)]:
            output_rows.append(
                {
                    "method": method,
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
                    "HBM_read_bytes_est": row.get("HBM_read_bytes_est"),
                    "HBM_write_bytes_est": row.get("HBM_write_bytes_est"),
                    "device": row.get("device"),
                    "gpu_name": row.get("gpu_name"),
                    "speedup_vs_baseline": speedup if is_fused else "",
                    "HBM_read_reduction_pct": hbm_reduction if is_fused else "",
                }
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
