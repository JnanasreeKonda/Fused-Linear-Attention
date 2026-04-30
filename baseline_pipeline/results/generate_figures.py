"""
results/generate_figures.py — Generate the profiling and Phase 3 figure set.

Expected inputs
---------------
- results/comparison_table.csv
- results/occupancy_sweep.csv          (optional but recommended)
- results/endtoend_timing.csv          (optional, used for annotations only)

Generated figures
-----------------
- results/figures/nsight_timeline_comparison.png
- results/figures/hbm_bandwidth.png
- results/figures/wall_time_speedup.png
- results/figures/speedup.png
- results/figures/occupancy_vs_tile_tradeoff.png
- results/figures/occupancy_vs_tile.png
- results/figures/kernel_count.png
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

COLOR_BASELINE = "#326FA8"
COLOR_FUSED = "#D95F02"
COLOR_NEUTRAL = "#7F7F7F"


def safe_float(row, col):
    value = row.get(col, "")
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_comparison_table():
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run results/merge_comparison.py first.")

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
        sys.exit("ERROR: no matching seq_len rows found in comparison_table.csv")
    return seq_lens, baseline, fused


def comparison_is_valid(baseline_row, fused_row):
    return (fused_row.get("comparison_status") or "").strip().lower() == "comparable"


def load_optional_rows(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def estimate_hbm_bytes(method: str, seq_len: int):
    batch_size = 1
    n_heads = 8
    embed_dim = 512
    d_head = 64
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


def plot_timeline_comparison(seq_lens, baseline, fused):
    ref_seq = 512 if 512 in baseline and 512 in fused else seq_lens[len(seq_lens) // 2]
    baseline_time = safe_float(baseline[ref_seq], "per_iter_us") or 0.0
    fused_time_raw = safe_float(fused[ref_seq], "per_iter_us") or 0.0
    comparable = comparison_is_valid(baseline[ref_seq], fused[ref_seq])
    fused_time = fused_time_raw if comparable else max(baseline_time * 0.65, 1.0)

    qkv_span = baseline_time * 0.45
    gap_span = baseline_time * 0.10
    attn_span = baseline_time * 0.45

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.broken_barh(
        [(0.0, qkv_span), (qkv_span + gap_span, attn_span)],
        (22, 8),
        facecolors=[COLOR_BASELINE, COLOR_BASELINE],
        edgecolors="black",
    )
    ax.broken_barh(
        [(0.0, fused_time)],
        (8, 8),
        facecolors=[COLOR_FUSED],
        edgecolors="black",
    )

    ax.text(qkv_span / 2, 26, "QKV projection", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(qkv_span + gap_span + attn_span / 2, 26, "SDPA", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(fused_time / 2, 12, "Fused QKV + attention", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.annotate(
        "HBM round-trip gap",
        xy=(qkv_span + gap_span / 2, 26),
        xytext=(qkv_span + gap_span / 2, 35),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": COLOR_NEUTRAL},
        color=COLOR_NEUTRAL,
        fontsize=9,
    )
    ax.set_yticks([12, 26])
    ax.set_yticklabels(["Fused", "Baseline"])
    ax.set_xlabel("Per-iteration timeline (schematic, microseconds)")
    ax.set_title(f"Kernel Timeline Comparison at seq_len={ref_seq}")
    ax.set_xlim(0, max(baseline_time, fused_time) * 1.15 if max(baseline_time, fused_time) > 0 else 1.0)
    ax.grid(axis="x", alpha=0.25)
    note = "Generated schematic from comparison_table.csv"
    if not comparable:
        note += " | fused width not to scale: profiling runs are not directly comparable"
    ax.text(0.99, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=COLOR_NEUTRAL)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "nsight_timeline_comparison.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hbm_bandwidth(seq_lens, baseline, fused):
    baseline_read_gb = []
    fused_read_gb = []
    baseline_write_gb = []
    fused_write_gb = []

    for seq_len in seq_lens:
        b_read = safe_float(baseline[seq_len], "HBM_read_bytes_est")
        b_write = safe_float(baseline[seq_len], "HBM_write_bytes_est")
        f_read = safe_float(fused[seq_len], "HBM_read_bytes_est")
        f_write = safe_float(fused[seq_len], "HBM_write_bytes_est")

        if b_read is None or b_write is None:
            b_read, b_write = estimate_hbm_bytes("baseline_unfused", seq_len)
        if f_read is None or f_write is None:
            f_read, f_write = estimate_hbm_bytes("fused_kernel", seq_len)

        baseline_read_gb.append(b_read / 1e9)
        fused_read_gb.append(f_read / 1e9)
        baseline_write_gb.append(b_write / 1e9)
        fused_write_gb.append(f_write / 1e9)

    x = np.arange(len(seq_lens))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(x - width / 2, baseline_read_gb, width, label="Baseline", color=COLOR_BASELINE)
    axes[0].bar(x + width / 2, fused_read_gb, width, label="Fused", color=COLOR_FUSED)
    axes[0].set_title("HBM Read Traffic")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Read bytes (GB)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in seq_lens])
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, baseline_write_gb, width, label="Baseline", color=COLOR_BASELINE)
    axes[1].bar(x + width / 2, fused_write_gb, width, label="Fused", color=COLOR_FUSED)
    axes[1].set_title("HBM Write Traffic")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Write bytes (GB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in seq_lens])
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hbm_bandwidth.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wall_time_speedup(seq_lens, baseline, fused):
    baseline_times = []
    fused_times = []
    speedups = []
    comparables = []

    for seq_len in seq_lens:
        b_time = safe_float(baseline[seq_len], "per_iter_us")
        f_time = safe_float(fused[seq_len], "per_iter_us")
        comparable = comparison_is_valid(baseline[seq_len], fused[seq_len])
        baseline_times.append(b_time)
        fused_times.append(f_time)
        comparables.append(comparable)
        speedups.append((b_time / f_time) if (comparable and b_time and f_time and f_time > 0) else None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(seq_lens, baseline_times, "o-", color=COLOR_BASELINE, linewidth=2, label="Baseline")
    axes[0].plot(seq_lens, fused_times, "s--", color=COLOR_FUSED, linewidth=2, label="Fused")
    axes[0].set_title("Absolute Latency")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Per-iteration latency (us)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(seq_lens)
    axes[0].set_xticklabels([str(v) for v in seq_lens])
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    valid_x = [seq for seq, spd in zip(seq_lens, speedups) if spd is not None]
    valid_y = [spd for spd in speedups if spd is not None]
    if valid_x:
        axes[1].plot(valid_x, valid_y, "o-", color=COLOR_FUSED, linewidth=2)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No directly comparable fused/baseline profiling runs.\nRun both on the same CUDA node with the compiled kernel.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            color=COLOR_NEUTRAL,
            fontsize=10,
        )
    axes[1].axhline(1.0, linestyle="--", linewidth=1, color=COLOR_NEUTRAL)
    axes[1].set_title("Speedup vs Baseline")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Speedup")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(seq_lens)
    axes[1].set_xticklabels([str(v) for v in seq_lens])
    axes[1].grid(alpha=0.3)

    if not all(comparables):
        axes[0].text(
            0.02,
            0.98,
            "Mixed profiling modes detected.\nAbsolute latencies shown for reference only.",
            ha="left",
            va="top",
            transform=axes[0].transAxes,
            fontsize=9,
            color=COLOR_NEUTRAL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=COLOR_NEUTRAL),
        )

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "wall_time_speedup.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "speedup.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_vs_tile_tradeoff(occupancy_rows):
    if not occupancy_rows:
        print("[figures] occupancy_sweep.csv not found; skipping occupancy figure.")
        return

    grouped = {}
    for row in occupancy_rows:
        tile_size = int(row["tile_size"])
        grouped.setdefault(tile_size, []).append(row)

    tile_sizes = sorted(grouped)
    theoretical_blocks = []
    measured_occupancy = []
    measured_latency = []

    for tile_size in tile_sizes:
        rows = grouped[tile_size]
        blocks = [safe_float(r, "theoretical_max_blocks_SM") for r in rows]
        occ = [safe_float(r, "SM_occupancy_pct") for r in rows]
        wall = [safe_float(r, "wall_time_ms") for r in rows]

        valid_blocks = [v for v in blocks if v is not None]
        valid_occ = [v for v in occ if v is not None]
        valid_wall = [v for v in wall if v is not None]

        theoretical_blocks.append(sum(valid_blocks) / len(valid_blocks) if valid_blocks else 0.0)
        measured_occupancy.append(sum(valid_occ) / len(valid_occ) if valid_occ else None)
        measured_latency.append(sum(valid_wall) / len(valid_wall) if valid_wall else None)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.bar(
        np.arange(len(tile_sizes)),
        theoretical_blocks,
        width=0.55,
        color=COLOR_BASELINE,
        alpha=0.8,
        label="Theoretical max blocks / SM",
    )
    ax1.set_xlabel("Tile size")
    ax1.set_ylabel("Blocks per SM")
    ax1.set_xticks(np.arange(len(tile_sizes)))
    ax1.set_xticklabels([str(v) for v in tile_sizes])
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    valid_occ_x = [i for i, v in enumerate(measured_occupancy) if v is not None]
    valid_occ_y = [v for v in measured_occupancy if v is not None]
    if valid_occ_x:
        ax2.plot(valid_occ_x, valid_occ_y, "o-", color=COLOR_FUSED, linewidth=2, label="Measured occupancy (%)")
        ax2.set_ylabel("Measured occupancy (%)")
    else:
        valid_wall_x = [i for i, v in enumerate(measured_latency) if v is not None]
        valid_wall_y = [v for v in measured_latency if v is not None]
        if valid_wall_x:
            ax2.plot(valid_wall_x, valid_wall_y, "s--", color=COLOR_FUSED, linewidth=2, label="Measured wall time (ms)")
            ax2.set_ylabel("Measured wall time (ms)")
        else:
            ax2.set_ylabel("Measured occupancy / latency")
            ax2.text(
                0.5,
                0.5,
                "Theoretical occupancy only.\nPopulate occupancy_sweep.csv with measured values from Greene/NSight Compute.",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                color=COLOR_NEUTRAL,
                fontsize=9,
            )

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title("Occupancy vs Tile-Size Trade-off")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "occupancy_vs_tile_tradeoff.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "occupancy_vs_tile.png"), dpi=180, bbox_inches="tight")
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
    fig.savefig(os.path.join(FIGURES_DIR, "kernel_count.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    seq_lens, baseline, fused = load_comparison_table()
    occupancy_rows = load_optional_rows("occupancy_sweep.csv")

    plot_timeline_comparison(seq_lens, baseline, fused)
    plot_hbm_bandwidth(seq_lens, baseline, fused)
    plot_wall_time_speedup(seq_lens, baseline, fused)
    plot_occupancy_vs_tile_tradeoff(occupancy_rows)
    plot_kernel_count(seq_lens)
    print(f"[figures] Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
