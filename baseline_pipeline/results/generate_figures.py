"""
results/generate_figures.py — Generate profiling figures from canonical CSVs.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_BASELINE = "#4878CF"
COLOR_FUSED = "#E87722"


def safe_float(row, col):
    v = row.get(col, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_comparison_table():
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")

    baseline = {}
    fused = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq_len = int(row["seq_len"])
            if row["method"] == "baseline_unfused":
                baseline[seq_len] = row
            elif row["method"] == "fused_kernel":
                fused[seq_len] = row

    seq_lens = sorted(set(baseline) & set(fused))
    if not seq_lens:
        sys.exit("ERROR: no matching seq_len rows in comparison_table.csv")
    return seq_lens, baseline, fused


def load_occupancy_sweep():
    path = os.path.join(RESULTS_DIR, "occupancy_sweep.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_hbm_bandwidth(seq_lens, baseline, fused):
    b_vals = []
    f_vals = []
    for seq_len in seq_lens:
        b_hbm = safe_float(baseline[seq_len], "HBM_read_bytes_est")
        f_hbm = safe_float(fused[seq_len], "HBM_read_bytes_est")

        if b_hbm is None:
            B, H, D, d = 1, 8, 512, 64
            b_hbm = (B * seq_len * D + 3 * B * H * seq_len * d * 2) * 4
        if f_hbm is None:
            B, H, D, d = 1, 8, 512, 64
            f_hbm = (B * seq_len * D + 3 * D * H * d) * 4

        b_vals.append(b_hbm / 1e9)
        f_vals.append(f_hbm / 1e9)

    x = np.arange(len(seq_lens))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, b_vals, width, label="Baseline", color=COLOR_BASELINE)
    ax.bar(x + width / 2, f_vals, width, label="Fused", color=COLOR_FUSED)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("HBM reads (GB)")
    ax.set_title("HBM Read Traffic")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hbm_bandwidth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(seq_lens, baseline, fused):
    speedups = []
    baseline_times = []
    fused_times = []
    for seq_len in seq_lens:
        b_t = safe_float(baseline[seq_len], "per_iter_us")
        f_t = safe_float(fused[seq_len], "per_iter_us")
        speedups.append((b_t / f_t) if (b_t and f_t and f_t > 0) else None)
        baseline_times.append(b_t)
        fused_times.append(f_t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    valid_x = [n for n, s in zip(seq_lens, speedups) if s is not None]
    valid_s = [s for s in speedups if s is not None]
    if valid_x:
        ax.plot(valid_x, valid_s, "o-", color=COLOR_FUSED, linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Speedup")
    ax.set_title("Wall-Time Speedup")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if all(t is not None for t in baseline_times):
        ax2.plot(seq_lens, baseline_times, "o-", color=COLOR_BASELINE, linewidth=2, label="Baseline")
    if any(t is not None for t in fused_times):
        f_x = [n for n, t in zip(seq_lens, fused_times) if t is not None]
        f_y = [t for t in fused_times if t is not None]
        ax2.plot(f_x, f_y, "s--", color=COLOR_FUSED, linewidth=2, label="Fused")
    ax2.set_xlabel("Sequence length N")
    ax2.set_ylabel("Per-iteration latency (us)")
    ax2.set_title("Absolute Latency")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(seq_lens)
    ax2.set_xticklabels([str(n) for n in seq_lens])
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "speedup.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_vs_tile(sweep_rows):
    if not sweep_rows:
        return

    rows_512 = [r for r in sweep_rows if int(r["seq_len"]) == 512]
    if not rows_512:
        return

    tile_sizes = [int(r["tile_size"]) for r in rows_512]
    shmem_kb = [float(r["shmem_per_block_KB"]) for r in rows_512]
    max_blocks = [int(r["theoretical_max_blocks_SM"]) for r in rows_512]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(tile_sizes, shmem_kb, color=COLOR_FUSED)
    axes[0].set_xlabel("Tile size")
    axes[0].set_ylabel("shmem per block (KB)")
    axes[0].set_title("Shared Memory by Tile Size")

    axes[1].bar(tile_sizes, max_blocks, color=COLOR_BASELINE)
    axes[1].set_xlabel("Tile size")
    axes[1].set_ylabel("Theoretical max blocks / SM")
    axes[1].set_title("Block Residency by Tile Size")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "occupancy_vs_tile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_count(seq_lens):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(seq_lens))
    width = 0.35
    ax.bar(x - width / 2, [2] * len(seq_lens), width, label="Baseline", color=COLOR_BASELINE)
    ax.bar(x + width / 2, [1] * len(seq_lens), width, label="Fused", color=COLOR_FUSED)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Kernel launches")
    ax.set_title("Kernel Count Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "kernel_count.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    seq_lens, baseline, fused = load_comparison_table()
    sweep_rows = load_occupancy_sweep()
    plot_hbm_bandwidth(seq_lens, baseline, fused)
    plot_speedup(seq_lens, baseline, fused)
    plot_occupancy_vs_tile(sweep_rows)
    plot_kernel_count(seq_lens)
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
