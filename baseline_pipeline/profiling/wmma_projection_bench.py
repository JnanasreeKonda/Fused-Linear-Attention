"""
profiling/wmma_projection_bench.py — benchmark the experimental WMMA projection prototype.

This isolates the X @ W projection stage, which has been the dominant
performance bottleneck in the fused attention kernel. The goal is to measure
whether a warp-cooperative Tensor Core implementation is promising enough to
justify a full fused-kernel rewrite around WMMA/Tensor Core tiles.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import torch

BASELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASELINE_ROOT)
for path in (BASELINE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import config
from kernel.load_wmma_proj import load_wmma_projection


def benchmark_once(X: torch.Tensor, W: torch.Tensor, warmup: int, timed: int):
    ext = load_wmma_projection()

    with torch.no_grad():
        for _ in range(warmup):
            _ = ext.forward(X, W)
            _ = X @ W
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(timed):
            _ = ext.forward(X, W)
    end.record()
    torch.cuda.synchronize()
    wmma_ms = start.elapsed_time(end)

    start.record()
    with torch.no_grad():
        for _ in range(timed):
            _ = X @ W
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end)

    out_wmma = ext.forward(X, W)
    out_torch = (X @ W).float()
    max_diff = float((out_wmma - out_torch).abs().max().item())
    mean_diff = float((out_wmma - out_torch).abs().mean().item())

    return {
        "wmma_us": (wmma_ms / timed) * 1e3,
        "torch_us": (torch_ms / timed) * 1e3,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="WMMA projection prototype benchmark")
    parser.add_argument("--out", default="results/wmma_projection_bench.csv")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--dtype", default="both", choices=["float16", "bfloat16", "both"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the WMMA projection benchmark.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    rows: List[dict] = []
    embed_dim = config.EMBED_DIM_BENCH
    head_dim = config.D_HEAD

    dtypes = []
    if args.dtype == "both":
        dtypes = [torch.float16, torch.bfloat16]
    elif args.dtype == "float16":
        dtypes = [torch.float16]
    else:
        dtypes = [torch.bfloat16]

    for torch_dtype in dtypes:
        for seq_len in config.SEQ_LENGTHS:
            M = seq_len
            K = embed_dim
            N = head_dim
            X = torch.randn(M, K, device=device, dtype=torch_dtype)
            W = (torch.randn(K, N, device=device, dtype=torch.float32) * 0.02).to(torch_dtype)

            result = benchmark_once(X, W, args.warmup, args.timed)
            row = {
                "dtype": str(torch_dtype).replace("torch.", ""),
                "seq_len": seq_len,
                "M": M,
                "K": K,
                "N": N,
                "wmma_us": round(result["wmma_us"], 4),
                "torch_us": round(result["torch_us"], 4),
                "speedup_vs_torch": round(result["torch_us"] / result["wmma_us"], 4),
                "max_abs_diff": result["max_abs_diff"],
                "mean_abs_diff": result["mean_abs_diff"],
            }
            rows.append(row)
            print(
                f"dtype={row['dtype']:<8} seq_len={seq_len:>5}  "
                f"wmma={row['wmma_us']:8.2f} us  "
                f"torch={row['torch_us']:8.2f} us  "
                f"speedup={row['speedup_vs_torch']:6.3f}x  "
                f"max_diff={row['max_abs_diff']:.3e}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[wmma_proj] Saved -> {args.out}")


if __name__ == "__main__":
    main()
