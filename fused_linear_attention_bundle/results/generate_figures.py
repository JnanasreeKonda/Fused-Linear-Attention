"""
results/generate_figures.py — M11 (Bhanuja Karumuru)

Generates all 4 result figures from the profiling CSVs.
Run after results/fused_profiling.csv and results/comparison_table.csv
are populated with real A100 numbers.

Usage:
    python results/generate_figures.py

Outputs (all in results/figures/):
    hbm_bandwidth.png       — HBM read traffic: baseline vs fused bar chart
    speedup.png             — Wall-time speedup vs sequence length
    occupancy_vs_tile.png   — SM occupancy + wall-time vs tile size at N=512
    kernel_count.png        — Kernel count comparison (2 vs 1)

Note: nsight_timeline.png must be captured manually as a screenshot from
      NSight Systems GUI (results/traces/fused/fused_nsys.nsys-rep) and saved
      to results/figures/nsight_timeline.png.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── consistent visual style ────────────────────────────────────────────────
COLOR_BASELINE = "#4878CF"   # blue
COLOR_FUSED    = "#E87722"   # orange
FIGSIZE_WIDE   = (8, 4)
FIGSIZE_SQ     = (6, 5)


def load_comparison_table():
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run merge_comparison.py first.")

    baseline, fused = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            N = int(row["seq_len"])
            if row["method"] == "baseline_unfused":
                baseline[N] = row
            elif row["method"] == "fused_kernel":
                fused[N] = row

    seq_lens = sorted(set(baseline) & set(fused))
    if not seq_lens:
        sys.exit("ERROR: no matching seq_len rows in comparison_table.csv")
    return seq_lens, baseline, fused


def safe_float(row, col):
    v = row.get(col, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_occupancy_sweep():
    path = os.path.join(RESULTS_DIR, "occupancy_sweep.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: HBM bandwidth bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_hbm_bandwidth(seq_lens, baseline, fused):
    """
    Bar chart of HBM_read_bytes_est (GB) for baseline vs fused at each seq_len.
    If NSight Compute exact values are not yet in the CSV, falls back to the
    theoretical estimate computed in fused_bench.py.
    """
    b_vals, f_vals = [], []
    for N in seq_lens:
        b_hbm = safe_float(baseline[N], "HBM_read_bytes_est")
        f_hbm = safe_float(fused[N],    "HBM_read_bytes_est")

        # Fallback: compute theoretical HBM reads
        # Baseline: X (read once for QKV proj) + Q,K,V written to HBM + Q,K,V read for SDPA
        #   = X*1 + QKV*2 (write) + QKV*1 (read) = X + 3*QKV total if we count all
        # Use peak_alloc as proxy only if no explicit value
        if b_hbm is None:
            B, H, D, d = 1, 8, 512, 64
            b_hbm = (B * N * D + 3 * B * H * N * d * 2) * 4   # X + QKV roundtrip
        if f_hbm is None:
            B, H, D, d = 1, 8, 512, 64
            f_hbm = (B * N * D + 3 * D * H * d) * 4            # X + weights only

        b_vals.append(b_hbm / 1e9)
        f_vals.append(f_hbm / 1e9)

    x      = np.arange(len(seq_lens))
    width  = 0.35
    labels = [str(N) for N in seq_lens]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars_b = ax.bar(x - width/2, b_vals, width, label="Baseline (unfused)",
                    color=COLOR_BASELINE, edgecolor="white", linewidth=0.5)
    bars_f = ax.bar(x + width/2, f_vals, width, label="Fused kernel",
                    color=COLOR_FUSED,    edgecolor="white", linewidth=0.5)

    # Annotate reduction % on fused bars
    for i, (b, f) in enumerate(zip(b_vals, f_vals)):
        if b > 0:
            pct = (1 - f / b) * 100
            ax.text(x[i] + width/2, f + max(b_vals)*0.01,
                    f"−{pct:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=COLOR_FUSED, fontweight="bold")

    ax.set_xlabel("Sequence length N", fontsize=11)
    ax.set_ylabel("HBM reads (GB)", fontsize=11)
    ax.set_title("HBM Read Traffic: Baseline vs Fused Kernel", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "hbm_bandwidth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Wall-time speedup vs sequence length
# ─────────────────────────────────────────────────────────────────────────────

def plot_speedup(seq_lens, baseline, fused):
    speedups, b_times, f_times = [], [], []
    for N in seq_lens:
        b_t = safe_float(baseline[N], "per_iter_us")
        f_t = safe_float(fused[N],    "per_iter_us")
        speedups.append((b_t / f_t) if (b_t and f_t and f_t > 0) else None)
        b_times.append(b_t)
        f_times.append(f_t)

    # Check if we have real speedup data
    has_data = any(s is not None for s in speedups)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: speedup
    ax = axes[0]
    xs = list(range(len(seq_lens)))
    if has_data:
        valid_x = [x for x, s in zip(xs, speedups) if s is not None]
        valid_s = [s for s in speedups if s is not None]
        ax.plot([seq_lens[i] for i in valid_x], valid_s,
                "o-", color=COLOR_FUSED, linewidth=2, markersize=7, label="Speedup")
        for x, s in zip([seq_lens[i] for i in valid_x], valid_s):
            ax.annotate(f"{s:.2f}×", (x, s), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=COLOR_FUSED)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "Run fused_bench.py first\nto populate data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")

    ax.set_xlabel("Sequence length N", fontsize=11)
    ax.set_ylabel("Speedup (baseline µs / fused µs)", fontsize=11)
    ax.set_title("Wall-Time Speedup: Fused vs Baseline", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(N) for N in seq_lens])
    ax.grid(alpha=0.3)
    if has_data:
        ax.legend()

    # Right: absolute times
    ax2 = axes[1]
    if has_data and all(t is not None for t in b_times):
        ax2.plot(seq_lens, b_times, "o-", color=COLOR_BASELINE, linewidth=2,
                 markersize=6, label="Baseline")
    if has_data and any(t is not None for t in f_times):
        f_x = [N for N, t in zip(seq_lens, f_times) if t is not None]
        f_y = [t for t in f_times if t is not None]
        ax2.plot(f_x, f_y, "s--", color=COLOR_FUSED, linewidth=2,
                 markersize=6, label="Fused kernel")
    if not has_data:
        ax2.text(0.5, 0.5, "Run fused_bench.py first",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="gray")

    ax2.set_xlabel("Sequence length N", fontsize=11)
    ax2.set_ylabel("Per-iteration latency (µs)", fontsize=11)
    ax2.set_title("Absolute Latency Comparison", fontsize=12)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(seq_lens)
    ax2.set_xticklabels([str(N) for N in seq_lens])
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: SM occupancy + throughput vs tile size at N=512
# ─────────────────────────────────────────────────────────────────────────────

def plot_occupancy_vs_tile(sweep_rows):
    if sweep_rows is None:
        print("  Skipping occupancy_vs_tile.png: occupancy_sweep.csv not found")
        return

    # Filter to N=512 rows
    rows_512 = [r for r in sweep_rows if int(r["seq_len"]) == 512]
    if not rows_512:
        print("  Skipping occupancy_vs_tile.png: no N=512 rows in sweep CSV")
        return

    tile_sizes = [int(r["tile_size"]) for r in rows_512]
    shmem_kb   = [float(r["shmem_per_block_KB"]) for r in rows_512]
    max_blocks = [int(r["theoretical_max_blocks_SM"]) for r in rows_512]

    # Try to get real occupancy and wall-time from CSV (filled after NSight runs)
    occupancy = [safe_float(r, "SM_occupancy_pct") for r in rows_512]
    wall_times = [safe_float(r, "wall_time_ms") for r in rows_512]
    has_occ  = any(o is not None for o in occupancy)
    has_time = any(t is not None for t in wall_times)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: shared memory usage
    ax = axes[0]
    bars = ax.bar(range(len(tile_sizes)), shmem_kb,
                  color=[COLOR_FUSED if ts == 64 else COLOR_BASELINE for ts in tile_sizes],
                  edgecolor="white")
    ax.set_xticks(range(len(tile_sizes)))
    ax.set_xticklabels([str(t) for t in tile_sizes])
    ax.set_xlabel("Tile size T", fontsize=11)
    ax.set_ylabel("shmem per block (KB)", fontsize=11)
    ax.set_title("Shared Memory Usage", fontsize=12)
    ax.axhline(164, color="red", linestyle="--", linewidth=1, alpha=0.5, label="164 KB limit")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (v, ts) in enumerate(zip(shmem_kb, tile_sizes)):
        ax.text(i, v + 2, f"{v:.0f}KB", ha="center", va="bottom", fontsize=9)

    # Panel 2: theoretical max blocks/SM
    ax2 = axes[1]
    ax2.bar(range(len(tile_sizes)), max_blocks,
            color=[COLOR_FUSED if ts == 64 else COLOR_BASELINE for ts in tile_sizes],
            edgecolor="white")
    ax2.set_xticks(range(len(tile_sizes)))
    ax2.set_xticklabels([str(t) for t in tile_sizes])
    ax2.set_xlabel("Tile size T", fontsize=11)
    ax2.set_ylabel("Max concurrent blocks / SM", fontsize=11)
    ax2.set_title("Theoretical SM Occupancy (blocks)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(max_blocks):
        ax2.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=10)

    # Panel 3: measured wall-time or occupancy
    ax3 = axes[2]
    if has_time:
        valid_t = [(ts, wt) for ts, wt in zip(tile_sizes, wall_times) if wt is not None]
        xs  = [v[0] for v in valid_t]
        ys  = [v[1] for v in valid_t]
        ax3.bar(range(len(xs)), ys,
                color=[COLOR_FUSED if ts == 64 else COLOR_BASELINE for ts in xs],
                edgecolor="white")
        ax3.set_xticks(range(len(xs)))
        ax3.set_xticklabels([str(t) for t in xs])
        ax3.set_ylabel("Wall time (ms)", fontsize=11)
        ax3.set_title("Measured Wall Time at N=512", fontsize=12)
        ax3.grid(axis="y", alpha=0.3)
    elif has_occ:
        valid_o = [(ts, o) for ts, o in zip(tile_sizes, occupancy) if o is not None]
        xs = [v[0] for v in valid_o]
        ys = [v[1] for v in valid_o]
        ax3.bar(range(len(xs)), ys,
                color=[COLOR_FUSED if ts == 64 else COLOR_BASELINE for ts in xs],
                edgecolor="white")
        ax3.set_xticks(range(len(xs)))
        ax3.set_xticklabels([str(t) for t in xs])
        ax3.set_ylabel("SM Occupancy (%)", fontsize=11)
        ax3.set_title("Measured SM Occupancy at N=512", fontsize=12)
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5,
                 "Fill results/occupancy_sweep.csv\nafter NSight Compute runs\n(Step 4 of run_bhanuja.sh)",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=10, color="gray", wrap=True)
        ax3.set_title("Measured Wall Time at N=512", fontsize=12)

    # Highlight selected tile
    selected = mpatches.Patch(color=COLOR_FUSED,    label="T=64 (selected)")
    other    = mpatches.Patch(color=COLOR_BASELINE,  label="Other tile sizes")
    fig.legend(handles=[selected, other], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle("Tile-Size Occupancy Sweep (N=512, A100)", fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "occupancy_vs_tile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Kernel count bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_kernel_count(seq_lens, baseline, fused):
    b_kernels = [int(baseline[N].get("kernel_count", 2) or 2) for N in seq_lens]
    f_kernels = [int(fused[N].get("kernel_count",    1) or 1) for N in seq_lens]

    x     = np.arange(len(seq_lens))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(x - width/2, b_kernels, width, label="Baseline (unfused)",
           color=COLOR_BASELINE, edgecolor="white")
    ax.bar(x + width/2, f_kernels, width, label="Fused kernel",
           color=COLOR_FUSED, edgecolor="white")

    ax.set_xlabel("Sequence length N", fontsize=11)
    ax.set_ylabel("CUDA kernel launches per forward pass", fontsize=11)
    ax.set_title("Kernel Count: Baseline vs Fused", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in seq_lens])
    ax.set_yticks([0, 1, 2, 3])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Labels on bars
    for xi, (b, f) in enumerate(zip(b_kernels, f_kernels)):
        ax.text(xi - width/2, b + 0.05, str(b), ha="center", va="bottom", fontsize=10)
        ax.text(xi + width/2, f + 0.05, str(f), ha="center", va="bottom", fontsize=10,
                color=COLOR_FUSED, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "kernel_count.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[figures] Loading data …")
    seq_lens, baseline, fused = load_comparison_table()
    sweep_rows = load_occupancy_sweep()
    print(f"[figures] seq_lens: {seq_lens}")
    print(f"[figures] Generating figures → {FIGURES_DIR}/\n")

    print("[figures] Figure 1: HBM bandwidth …")
    plot_hbm_bandwidth(seq_lens, baseline, fused)

    print("[figures] Figure 2: Speedup …")
    plot_speedup(seq_lens, baseline, fused)

    print("[figures] Figure 3: Occupancy vs tile size …")
    plot_occupancy_vs_tile(sweep_rows)

    print("[figures] Figure 4: Kernel count …")
    plot_kernel_count(seq_lens, baseline, fused)

    print("\n[figures] Done. Remaining manual step:")
    print("  Take a screenshot of NSight Systems side-by-side timeline and save as:")
    print("  results/figures/nsight_timeline.png")


if __name__ == "__main__":
    main()
