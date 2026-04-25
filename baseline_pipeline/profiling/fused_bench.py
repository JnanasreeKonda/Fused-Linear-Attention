"""
profiling/fused_bench.py — Canonical fused-kernel microbenchmark.

This mirrors `baseline_bench.py` but uses the root-level fused kernel loader.
It lives in `baseline_pipeline/` so both baseline and fused profiling scripts
share the same config and results directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASELINE_ROOT)
for path in (BASELINE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import config

try:
    import nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False


class FusedQKVAttentionSimulated(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.n_heads, self.d_head

        Q = (x @ self.Wq).view(B, N, H, d).transpose(1, 2)
        K = (x @ self.Wk).view(B, N, H, d).transpose(1, 2)
        V = (x @ self.Wv).view(B, N, H, d).transpose(1, 2)
        return F.scaled_dot_product_attention(Q, K, V)


class FusedQKVAttentionKernel(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

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
            B,
            self.n_heads,
            N,
            D,
            self.d_head,
        )


def benchmark_one(
    model: nn.Module,
    seq_len: int,
    embed_dim: int,
    batch_size: int,
    device: torch.device,
    warmup: int,
    timed: int,
) -> dict:
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if _HAS_NVTX:
        nvtx.push_range(f"fused_seq{seq_len}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(timed):
                _ = model(x)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        mem_stats = torch.cuda.memory_stats(device)
        peak_mb = mem_stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(timed):
                _ = model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        peak_mb = 0.0

    if _HAS_NVTX:
        nvtx.pop_range()
    per_iter_us = (elapsed_ms / timed) * 1e3

    d_head = embed_dim // config.N_HEADS_BENCH
    n_heads = config.N_HEADS_BENCH
    fp32_bytes = 4
    hbm_read = (batch_size * seq_len * embed_dim + 3 * embed_dim * n_heads * d_head) * fp32_bytes
    hbm_write = (batch_size * n_heads * seq_len * d_head) * fp32_bytes

    return {
        "method": "fused_kernel",
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "n_heads": n_heads,
        "batch_size": batch_size,
        "warmup_iters": warmup,
        "timed_iters": timed,
        "total_elapsed_ms": round(elapsed_ms, 4),
        "per_iter_us": round(per_iter_us, 4),
        "peak_alloc_mb": round(peak_mb, 4),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "kernel_count": 1,
        "HBM_read_bytes_est": hbm_read,
        "HBM_write_bytes_est": hbm_write,
    }


def build_occupancy_sweep() -> list[dict]:
    tile_sizes = [16, 32, 64, 128]
    seq_lens = config.SEQ_LENGTHS
    d_head = config.D_HEAD
    shmem_total_kb = 164.0

    rows = []
    for tile_size in tile_sizes:
        shmem_bytes = 3 * tile_size * (d_head + 1) * 4
        shmem_kb = shmem_bytes / 1024
        max_blocks = int(shmem_total_kb / shmem_kb) if shmem_kb > 0 else 0
        for seq_len in seq_lens:
            rows.append(
                {
                    "tile_size": tile_size,
                    "seq_len": seq_len,
                    "shmem_per_block_bytes": shmem_bytes,
                    "shmem_per_block_KB": round(shmem_kb, 2),
                    "theoretical_max_blocks_SM": max_blocks,
                    "wall_time_ms": "",
                    "SM_occupancy_pct": "",
                    "notes": "SELECTED" if tile_size == 64 else "",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fused attention microbenchmark")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", default="results/fused_profiling.csv")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=config.WARMUP_ITERS)
    parser.add_argument("--timed", type=int, default=config.TIMED_ITERS)
    args = parser.parse_args()

    device = torch.device("cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda")
    embed_dim = config.EMBED_DIM_BENCH
    n_heads = config.N_HEADS_BENCH
    batch_size = config.BATCH_BENCH
    seq_lens = [args.seq_len] if args.seq_len else config.SEQ_LENGTHS

    if args.simulate:
        model = FusedQKVAttentionSimulated(embed_dim, n_heads).to(device).eval()
        print("[fused_bench] Mode: PyTorch simulation")
    else:
        if device.type != "cuda":
            raise RuntimeError("Compiled fused kernel requires CUDA. Use --simulate for CPU validation.")
        model = FusedQKVAttentionKernel(embed_dim, n_heads).to(device).eval()
        dummy = torch.randn(1, 64, embed_dim, device=device)
        with torch.no_grad():
            _ = model(dummy)
        print("[fused_bench] Mode: compiled CUDA kernel")

    print(f"[fused_bench] Device    : {device}")
    if device.type == "cuda":
        print(f"[fused_bench] GPU       : {torch.cuda.get_device_name(device)}")
    print(f"[fused_bench] embed_dim : {embed_dim}  n_heads: {n_heads}  batch: {batch_size}")
    print(f"[fused_bench] warmup    : {args.warmup}  timed: {args.timed}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results/traces/fused", exist_ok=True)

    results = []
    for seq_len in seq_lens:
        print(f"  seq_len={seq_len:>5} ... ", end="", flush=True)
        row = benchmark_one(model, seq_len, embed_dim, batch_size, device, args.warmup, args.timed)
        results.append(row)
        print(
            f"per_iter={row['per_iter_us']:8.1f} us  |  "
            f"peak_alloc={row['peak_alloc_mb']:7.2f} MB  |  "
            f"kernels={row['kernel_count']}"
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[fused_bench] Saved -> {args.out}")

    sweep_path = "results/occupancy_sweep.csv"
    sweep_rows = build_occupancy_sweep()
    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"[fused_bench] Saved -> {sweep_path}")


if __name__ == "__main__":
    main()
