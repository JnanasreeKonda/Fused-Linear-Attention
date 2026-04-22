"""
profiling/fused_bench.py — Fused kernel NSight microbenchmark.
Owner: Bhanuja Karumuru  |  Milestone: M9  |  Phase 3

Mirrors baseline_bench.py EXACTLY:
  - Same embed_dim (config.EMBED_DIM_BENCH = 512)
  - Same n_heads   (config.N_HEADS_BENCH   = 8)
  - Same batch     (config.BATCH_BENCH     = 1)
  - Same seq_lens  (config.SEQ_LENGTHS     = [64, 128, 256, 512, 1024])
  - Same warmup / timed / CUDA Event timing protocol
  - Same CSV column names for merge_comparison.py

Two run modes
─────────────
1. With compiled CUDA kernel (requires Jnanasree's M8 handoff):
       python profiling/fused_bench.py

2. Simulation mode — pure PyTorch fused-equivalent for benchmarking logic
   (use this while waiting for M8, or to test on non-A100):
       python profiling/fused_bench.py --simulate

3. Full NSight Systems trace (A100 / Greene):
       nsys profile --trace=cuda,nvtx \\
            --output=results/traces/fused/fused_nsys \\
            python profiling/fused_bench.py

4. NSight Compute (per-kernel HBM bytes + occupancy, one seq_len at a time):
       ncu --set full \\
           --target-processes all \\
           --metrics l1tex__t_bytes,sm__warps_active.avg.pct_of_peak_sustained_active \\
           --export results/traces/fused/ncu_seq512 \\
           python profiling/fused_bench.py --seq_len 512

Outputs
───────
results/fused_profiling.csv     — wall-time row per seq_len
results/occupancy_sweep.csv     — tile-size × seq_len theoretical shmem table
                                  (fill SM_occupancy_pct from NSight Compute)
results/traces/fused/           — NSight trace files

CSV columns (match baseline_profiling.csv for merge_comparison.py):
    method, seq_len, embed_dim, n_heads, batch_size,
    warmup_iters, timed_iters,
    total_elapsed_ms, per_iter_us,
    peak_alloc_mb,
    device, gpu_name,
    kernel_count          (extra: 1 for fused vs 2+ for baseline)
    HBM_read_bytes_est    (extra: theoretical; NSight gives exact)
    HBM_write_bytes_est   (extra: theoretical)
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

try:
    import nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch simulation of the fused kernel
# (used when --simulate or when M8 kernel not yet compiled)
# ─────────────────────────────────────────────────────────────────────────────

class FusedQKVAttention_Simulated(nn.Module):
    """
    Simulates the fused kernel's behaviour using PyTorch ops.
    NOT the actual CUDA kernel — only used for benchmarking logic
    validation before Jnanasree's M8 handoff.

    Difference from baseline_bench.py's UnfusedQKVAttention:
      - Separate weight matrices Wq, Wk, Wv (no nn.Linear, no bias)
      - torch.matmul directly (closer to what the CUDA kernel computes)
      - No out_proj (kernel output is pre-projection; out_proj is separate)
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = embed_dim // n_heads
        self.embed_dim = embed_dim

        # Separate weight matrices — matches kernel interface
        # Wq/Wk/Wv: [embed_dim, embed_dim] = [D, H*d]
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → out: [B, H, N, d_head]"""
        B, N, D = x.shape
        H, d = self.n_heads, self.d_head

        # QKV projection: [B, N, D] @ [D, H*d] → [B, N, H*d]
        # Reshape to [B, H, N, d] for multi-head
        Q = (x @ self.Wq).view(B, N, H, d).transpose(1, 2)   # [B, H, N, d]
        K = (x @ self.Wk).view(B, N, H, d).transpose(1, 2)
        V = (x @ self.Wv).view(B, N, H, d).transpose(1, 2)

        # SDPA — in a true fused kernel this happens in the same kernel as QKV
        out = F.scaled_dot_product_attention(Q, K, V)          # [B, H, N, d]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused kernel wrapper (calls through load_kernel.py after M8 handoff)
# ─────────────────────────────────────────────────────────────────────────────

class FusedQKVAttention_Kernel(nn.Module):
    """Thin wrapper around the compiled CUDA extension."""

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

        self._kernel = None

    def _get_kernel(self):
        if self._kernel is None:
            from kernel.load_kernel import load_fused_kernel
            self._kernel = load_fused_kernel()
        return self._kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return self._get_kernel().forward(
            x.contiguous(),
            self.Wq.contiguous(),
            self.Wk.contiguous(),
            self.Wv.contiguous(),
            B, self.n_heads, N, D, self.d_head
        )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark one seq_len
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    model: nn.Module,
    seq_len: int,
    embed_dim: int,
    batch_size: int,
    device: torch.device,
    warmup: int,
    timed: int,
    method_label: str,
) -> dict:
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize(device)

    # Timed with CUDA Events (same protocol as baseline_bench.py)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    if _HAS_NVTX:
        nvtx.push_range(f"fused_seq{seq_len}")

    start.record()
    with torch.no_grad():
        for _ in range(timed):
            _ = model(x)
    end.record()

    if _HAS_NVTX:
        nvtx.pop_range()

    torch.cuda.synchronize(device)

    elapsed_ms  = start.elapsed_time(end)
    per_iter_us = (elapsed_ms / timed) * 1e3
    mem_stats   = torch.cuda.memory_stats(device)
    peak_mb     = mem_stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2)

    # Theoretical HBM bytes (exact from NSight Compute; this is an estimate)
    # Fused reads: X[B,N,D] + Wq[D,H*d] + Wk[D,H*d] + Wv[D,H*d]
    # Fused writes: Out[B,H,N,d]
    d_head    = embed_dim // config.N_HEADS_BENCH
    n_heads   = config.N_HEADS_BENCH
    fp32_bytes = 4
    hbm_read  = (batch_size * seq_len * embed_dim
                 + 3 * embed_dim * n_heads * d_head) * fp32_bytes
    hbm_write = (batch_size * n_heads * seq_len * d_head) * fp32_bytes

    return {
        "method":             method_label,
        "seq_len":            seq_len,
        "embed_dim":          embed_dim,
        "n_heads":            n_heads,
        "batch_size":         batch_size,
        "warmup_iters":       warmup,
        "timed_iters":        timed,
        "total_elapsed_ms":   round(elapsed_ms,  4),
        "per_iter_us":        round(per_iter_us, 4),
        "peak_alloc_mb":      round(peak_mb,     4),
        "device":             str(device),
        "gpu_name":           torch.cuda.get_device_name(device),
        "kernel_count":       1,                          # fused = 1 kernel
        "HBM_read_bytes_est": hbm_read,
        "HBM_write_bytes_est": hbm_write,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy sweep table (theoretical — fill from NSight Compute after M9 runs)
# ─────────────────────────────────────────────────────────────────────────────

def build_occupancy_sweep() -> list:
    """
    Pre-computes theoretical shmem per block and max blocks/SM for each
    (tile_size, seq_len) combination.

    After running NSight Compute:
        ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ...
    fill SM_occupancy_pct and wall_time_ms columns manually.
    """
    tile_sizes = [16, 32, 64, 128]
    seq_lens   = config.SEQ_LENGTHS
    d_head     = config.D_HEAD    # 64
    shmem_total_kb = 164.0        # A100 per-SM

    rows = []
    for T in tile_sizes:
        shmem_bytes = 3 * T * (d_head + 1) * 4   # sQ + sK + sV only (sO in registers)
        shmem_kb    = shmem_bytes / 1024
        max_blocks  = int(shmem_total_kb / shmem_kb) if shmem_kb > 0 else 0
        for N in seq_lens:
            rows.append({
                "tile_size":                T,
                "seq_len":                  N,
                "shmem_per_block_bytes":    shmem_bytes,
                "shmem_per_block_KB":       round(shmem_kb, 2),
                "theoretical_max_blocks_SM": max_blocks,
                "wall_time_ms":             "",    # fill from NSight run
                "SM_occupancy_pct":         "",    # fill from NSight Compute
                "notes": "SELECTED" if T == 64 else "",
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fused attention microbenchmark (M9)")
    parser.add_argument("--simulate",   action="store_true",
                        help="Use PyTorch simulation (no compiled CUDA kernel needed)")
    parser.add_argument("--out",        default="results/fused_profiling.csv")
    parser.add_argument("--seq_len",    type=int, default=None,
                        help="Single seq_len (for ncu single-kernel capture)")
    parser.add_argument("--warmup",     type=int, default=config.WARMUP_ITERS)
    parser.add_argument("--timed",      type=int, default=config.TIMED_ITERS)
    args = parser.parse_args()

    assert torch.cuda.is_available(), (
        "CUDA not available. Run on Greene A100 node.\n"
        "  srun --gres=gpu:1 --pty bash\n"
        "  module load cuda/12.1"
    )

    device     = torch.device("cuda")
    embed_dim  = config.EMBED_DIM_BENCH    # 512
    n_heads    = config.N_HEADS_BENCH      # 8
    batch_size = config.BATCH_BENCH        # 1
    seq_lens   = [args.seq_len] if args.seq_len else config.SEQ_LENGTHS

    gpu_name = torch.cuda.get_device_name(device)
    print(f"[fused_bench] GPU       : {gpu_name}")
    print(f"[fused_bench] embed_dim : {embed_dim}  n_heads: {n_heads}  batch: {batch_size}")
    print(f"[fused_bench] warmup    : {args.warmup}  timed: {args.timed}")

    if args.simulate:
        model        = FusedQKVAttention_Simulated(embed_dim, n_heads).to(device).eval()
        method_label = "fused_simulated"
        print("[fused_bench] Mode: PyTorch simulation (--simulate)")
    else:
        try:
            model        = FusedQKVAttention_Kernel(embed_dim, n_heads).to(device).eval()
            # Trigger compile now rather than inside the timed loop
            dummy = torch.randn(1, 64, embed_dim, device=device)
            with torch.no_grad():
                _ = model(dummy)
            method_label = "fused_cuda_kernel"
            print("[fused_bench] Mode: compiled CUDA kernel")
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"[fused_bench] WARNING: kernel not available ({e})")
            print("[fused_bench] Falling back to --simulate. Re-run without --simulate after M8.")
            model        = FusedQKVAttention_Simulated(embed_dim, n_heads).to(device).eval()
            method_label = "fused_simulated"

    print()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results/traces/fused", exist_ok=True)

    results = []
    for seq_len in seq_lens:
        print(f"  seq_len={seq_len:>5} … ", end="", flush=True)
        row = benchmark_one(
            model, seq_len, embed_dim, batch_size, device,
            warmup=args.warmup, timed=args.timed,
            method_label=method_label,
        )
        results.append(row)
        print(
            f"per_iter={row['per_iter_us']:8.1f} µs  |  "
            f"peak_alloc={row['peak_alloc_mb']:7.2f} MB  |  "
            f"kernels={row['kernel_count']}"
        )

    # Write fused_profiling.csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[fused_bench] Saved → {args.out}")

    # Write occupancy_sweep.csv
    sweep_path = "results/occupancy_sweep.csv"
    sweep_rows = build_occupancy_sweep()
    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"[fused_bench] Saved → {sweep_path}")
    print("              (fill SM_occupancy_pct and wall_time_ms from NSight Compute)")

    # Print NSight instructions
    print()
    print("[fused_bench] ── NSight Systems trace ─────────────────────────────────")
    print("  nsys profile --trace=cuda,nvtx \\")
    print("       --output=results/traces/fused/fused_nsys \\")
    print("       python profiling/fused_bench.py")
    print()
    print("[fused_bench] ── NSight Compute (HBM bytes + occupancy) ────────────────")
    print("  for SEQ in 64 128 256 512 1024; do")
    print("    ncu --set full \\")
    print("        --metrics l1tex__t_bytes,sm__warps_active.avg.pct_of_peak_sustained_active \\")
    print("        --export results/traces/fused/ncu_seq${SEQ} \\")
    print("        python profiling/fused_bench.py --seq_len ${SEQ} --warmup 10 --timed 10")
    print("  done")
    print("[fused_bench] ────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
