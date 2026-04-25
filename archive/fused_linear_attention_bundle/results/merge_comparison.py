"""
results/merge_comparison.py — M11 (Bhanuja Karumuru)

Merges baseline_profiling.csv (Rithwik M3) and fused_profiling.csv (Bhanuja M9)
into a unified comparison_table.csv with speedup and HBM reduction columns.

Run after both CSVs are populated with real A100 numbers:
    python results/merge_comparison.py

Reads:
    results/baseline_profiling.csv   (columns from baseline_bench.py)
    results/fused_profiling.csv      (columns from fused_bench.py)

Writes:
    results/comparison_table.csv
"""

import csv
import os
import sys


def load_csv_keyed(path: str, key: str = "seq_len") -> dict:
    """Load CSV and return dict keyed by int(row[key])."""
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
    base_path  = os.path.join(os.path.dirname(__file__), "baseline_profiling.csv")
    fused_path = os.path.join(os.path.dirname(__file__), "fused_profiling.csv")
    out_path   = os.path.join(os.path.dirname(__file__), "comparison_table.csv")

    if not os.path.exists(base_path):
        sys.exit(f"ERROR: {base_path} not found. Run Rithwik's baseline_bench.py first (M3).")
    if not os.path.exists(fused_path):
        sys.exit(f"ERROR: {fused_path} not found. Run Bhanuja's fused_bench.py first (M9).")

    baseline = load_csv_keyed(base_path)
    fused    = load_csv_keyed(fused_path)
    seq_lens = sorted(set(baseline) & set(fused))

    if not seq_lens:
        sys.exit("ERROR: No matching seq_len values between baseline and fused CSVs.")

    output_rows = []
    for seq_len in seq_lens:
        b = baseline[seq_len]
        f = fused[seq_len]

        b_time   = safe_float(b, "per_iter_us")
        f_time   = safe_float(f, "per_iter_us")
        speedup  = round(b_time / f_time, 4) if (b_time and f_time and f_time > 0) else ""

        # HBM: baseline has no HBM_read_bytes_est (it's an unfused model with peak_alloc only).
        # The theoretical reduction comes from kernel analysis in DESIGN.md.
        # If fused_profiling.csv has HBM_read_bytes_est, compute reduction vs. unfused estimate.
        b_hbm = safe_float(b, "HBM_read_bytes_est")
        f_hbm = safe_float(f, "HBM_read_bytes_est")
        hbm_reduction = (
            round((1.0 - f_hbm / b_hbm) * 100, 2)
            if (b_hbm and f_hbm and b_hbm > 0)
            else ""
        )

        def get(row, col, default=""):
            return row.get(col, default)

        for method, row, is_fused in [("baseline_unfused", b, False), ("fused_kernel", f, True)]:
            output_rows.append({
                "method":                 method,
                "seq_len":                seq_len,
                "embed_dim":              get(row, "embed_dim", 512),
                "n_heads":                get(row, "n_heads",   8),
                "batch_size":             get(row, "batch_size", 1),
                "warmup_iters":           get(row, "warmup_iters"),
                "timed_iters":            get(row, "timed_iters"),
                "total_elapsed_ms":       get(row, "total_elapsed_ms"),
                "per_iter_us":            get(row, "per_iter_us"),
                "peak_alloc_mb":          get(row, "peak_alloc_mb"),
                "kernel_count":           get(row, "kernel_count", "2" if not is_fused else "1"),
                "HBM_read_bytes_est":     get(row, "HBM_read_bytes_est"),
                "HBM_write_bytes_est":    get(row, "HBM_write_bytes_est"),
                "device":                 get(row, "device"),
                "gpu_name":               get(row, "gpu_name"),
                "speedup_vs_baseline":    speedup  if is_fused else "",
                "HBM_read_reduction_pct": hbm_reduction if is_fused else "",
            })

    fieldnames = list(output_rows[0].keys())
    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved: {out_path}")
    print()
    print(f"{'seq_len':>8}  {'baseline µs':>14}  {'fused µs':>12}  {'speedup':>10}  {'HBM red%':>10}")
    print("-" * 62)
    for r in output_rows:
        if r["method"] == "fused_kernel":
            b_row = next(x for x in output_rows if x["seq_len"] == r["seq_len"] and x["method"] == "baseline_unfused")
            print(
                f"{r['seq_len']:>8}  "
                f"{str(b_row['per_iter_us']):>14}  "
                f"{str(r['per_iter_us']):>12}  "
                f"{str(r['speedup_vs_baseline']):>10}x  "
                f"{str(r['HBM_read_reduction_pct']):>8}%"
            )


if __name__ == "__main__":
    main()
