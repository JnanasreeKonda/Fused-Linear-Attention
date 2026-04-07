"""
profiling/baseline_bench.py — Baseline NSight microbenchmark.
Owner: Rithwik Amajala  |  Milestone: M3  |  Phase 1

Benchmarks the *unfused* QKV-projection + SDPA pipeline (standard PyTorch).
This is the two-kernel baseline that FusedLinearAttention is compared against.

Two run modes
─────────────
1. Wall-time only (any machine):
       python profiling/baseline_bench.py

2. Full NSight Systems trace (A100 / Greene node):
       nsys profile --trace=cuda,nvtx \\
            --output=results/traces/baseline/baseline \\
            python profiling/baseline_bench.py

   Convert to human-readable CSV after profiling:
       nsys stats --report=cuda_gpu_kern_sum results/traces/baseline/baseline.nsys-rep

Outputs
───────
results/baseline_profiling.csv      — wall-time + memory columns, one row per seq_len
results/traces/baseline/            — NSight trace files (when run under nsys)

CSV columns
───────────
method, seq_len, embed_dim, n_heads, batch_size,
warmup_iters, timed_iters,
total_elapsed_ms, per_iter_us,
peak_alloc_mb,
device, gpu_name
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Optional NVTX annotations (available on Greene; no-op otherwise)
try:
    import nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False


# ──────────────────────────────────────────────────────────────────────────────
# Unfused module — THE BASELINE
# ──────────────────────────────────────────────────────────────────────────────

class UnfusedQKVAttention(nn.Module):
    """
    Standard two-kernel pipeline:
      Kernel 1 — three separate nn.Linear  →  Q, K, V written to HBM
      Kernel 2 — F.scaled_dot_product_attention reads Q,K,V from HBM

    This is exactly what FusedLinearAttention replaces with a single kernel.
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = embed_dim // n_heads
        self.embed_dim = embed_dim

        # Three separate projections (unfused — writes QKV to HBM)
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # ── Kernel 1: QKV projection ─────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # ── Kernel 2: SDPA (reads QKV back from HBM) ─────────────────────────
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, D))


# ──────────────────────────────────────────────────────────────────────────────
# Per-sequence-length benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    model: nn.Module,
    seq_len: int,
    embed_dim: int,
    batch_size: int,
    device: torch.device,
    warmup: int,
    timed: int,
) -> dict:
    """
    Run `warmup` un-timed iterations, then `timed` iterations with CUDA Events.

    Notes on memory stats
    ─────────────────────
    torch.cuda.memory_stats gives *allocation* counts, not raw HBM read/write
    bytes.  True HBM bandwidth numbers (bytes_read / bytes_written) require
    NSight Compute (`ncu --metrics l1tex__t_bytes`).  The CSV column
    `peak_alloc_mb` is therefore an allocation-level proxy; the NSight trace
    is the authoritative source for HBM bandwidth.
    """
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

    # ── Warmup ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize(device)

    # ── Timed run ─────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    if _HAS_NVTX:
        nvtx.push_range(f"baseline_unfused_seq{seq_len}")

    start.record()
    with torch.no_grad():
        for _ in range(timed):
            _ = model(x)
    end.record()

    if _HAS_NVTX:
        nvtx.pop_range()

    torch.cuda.synchronize(device)

    elapsed_ms   = start.elapsed_time(end)
    per_iter_us  = (elapsed_ms / timed) * 1e3
    mem_stats    = torch.cuda.memory_stats(device)
    peak_mb      = mem_stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2)

    return {
        "method":          "baseline_unfused",
        "seq_len":         seq_len,
        "embed_dim":       embed_dim,
        "n_heads":         config.N_HEADS_BENCH,
        "batch_size":      batch_size,
        "warmup_iters":    warmup,
        "timed_iters":     timed,
        "total_elapsed_ms": round(elapsed_ms,  4),
        "per_iter_us":      round(per_iter_us, 4),
        "peak_alloc_mb":    round(peak_mb,     4),
        "device":          str(device),
        "gpu_name":        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline unfused-attention microbenchmark")
    parser.add_argument("--out",          default="results/baseline_profiling.csv")
    parser.add_argument("--seq-lengths",  nargs="+", type=int, default=config.SEQ_LENGTHS)
    parser.add_argument("--warmup",       type=int,  default=config.WARMUP_ITERS)
    parser.add_argument("--timed",        type=int,  default=config.TIMED_ITERS)
    parser.add_argument("--no-cuda",      action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    )

    if device.type != "cuda":
        print(
            "[bench] WARNING: CUDA not available.\n"
            "        Wall-time numbers on CPU are meaningless for this benchmark.\n"
            "        Run on a Greene A100 node for real profiling.\n"
        )
    else:
        print(f"[bench] GPU: {torch.cuda.get_device_name(device)}")

    embed_dim  = config.EMBED_DIM_BENCH
    n_heads    = config.N_HEADS_BENCH
    batch_size = config.BATCH_BENCH

    model = UnfusedQKVAttention(embed_dim, n_heads).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[bench] embed_dim={embed_dim}  n_heads={n_heads}  batch={batch_size}")
    print(f"[bench] warmup={args.warmup}  timed={args.timed}")
    print(f"[bench] Model parameters: {n_params:,}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results/traces/baseline", exist_ok=True)

    results = []
    for seq_len in args.seq_lengths:
        print(f"  seq_len={seq_len:>5} … ", end="", flush=True)
        row = benchmark_one(
            model, seq_len, embed_dim, batch_size, device,
            warmup=args.warmup, timed=args.timed
        )
        results.append(row)
        print(
            f"per_iter={row['per_iter_us']:8.1f} µs  |  "
            f"peak_alloc={row['peak_alloc_mb']:7.2f} MB"
        )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[bench] Results saved → {args.out}")
    print()
    print("[bench] ── NSight Systems trace command ──────────────────────────────")
    print(f"  nsys profile \\")
    print(f"    --trace=cuda,nvtx \\")
    print(f"    --output=results/traces/baseline/baseline \\")
    print(f"    python profiling/baseline_bench.py")
    print()
    print("[bench] ── NSight Compute (per-kernel HBM bytes) ─────────────────────")
    print(f"  ncu --metrics l1tex__t_bytes,sm__throughput \\")
    print(f"       --target-processes all \\")
    print(f"       --output results/traces/baseline/ncu_baseline \\")
    print(f"       python profiling/baseline_bench.py --timed 10")
    print("[bench] ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
