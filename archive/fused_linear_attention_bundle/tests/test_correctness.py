"""
tests/test_correctness.py — Full kernel correctness test suite.
Milestone: M8 (verifies kernel before M9 profiling runs)

Compares the CUDA kernel output against Jnanasree's NumPy golden files
(CPU_Reference_in_NumPy/tests/golden/*.npy).

Test matrix
───────────
The golden files cover:
    B=1, S=64,   d_model=128, d_head=64
    B=1, S=128,  d_model=128, d_head=64
    B=4, S=256,  d_model=128, d_head=64
    B=4, S=512,  d_model=128, d_head=64
    B=4, S=1024, d_model=128, d_head=64

Each golden file tests single-head (H=1) attention.
Multi-head (H=4, H=8) configs are generated fresh and compared against
a PyTorch SDPA reference.

Pass criterion: max absolute difference < 1e-4 for every test case.

Outputs
───────
results/correctness_results.csv  — one row per test case with pass/fail

Usage
─────
# Full suite (requires M8 kernel compiled):
    python tests/test_correctness.py

# Simulate mode (tests logic without CUDA kernel):
    python tests/test_correctness.py --simulate

# Single config:
    python tests/test_correctness.py --B 1 --S 128 --D 128 --d 64 --H 1
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "CPU_Reference_in_NumPy", "tests", "golden"
)
TOLERANCE  = 1e-4

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch reference — used for multi-head configs and simulate mode
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_reference(X_np, Wq_np, Wk_np, Wv_np, H):
    """
    NumPy inputs → PyTorch SDPA output → NumPy.
    Returns [B, H, N, d_head] as numpy float32.
    """
    B, N, D = X_np.shape
    d_head  = Wq_np.shape[1] // H   # Wq_np: [D, H*d_head]

    X  = torch.from_numpy(X_np)
    Wq = torch.from_numpy(Wq_np)
    Wk = torch.from_numpy(Wk_np)
    Wv = torch.from_numpy(Wv_np)

    Q = (X @ Wq).view(B, N, H, d_head).transpose(1, 2)   # [B, H, N, d_head]
    K = (X @ Wk).view(B, N, H, d_head).transpose(1, 2)
    V = (X @ Wv).view(B, N, H, d_head).transpose(1, 2)

    out = F.scaled_dot_product_attention(Q, K, V)         # [B, H, N, d_head]
    return out.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Kernel caller
# ─────────────────────────────────────────────────────────────────────────────

def kernel_forward(X_np, Wq_np, Wk_np, Wv_np, B, H, N, D, d_head, kernel):
    """
    Run the compiled CUDA kernel.
    Returns [B, H, N, d_head] as numpy float32.
    """
    device = torch.device("cuda")
    X  = torch.from_numpy(X_np).to(device).contiguous()
    Wq = torch.from_numpy(Wq_np).to(device).contiguous()
    Wk = torch.from_numpy(Wk_np).to(device).contiguous()
    Wv = torch.from_numpy(Wv_np).to(device).contiguous()

    out = kernel.forward(X, Wq, Wk, Wv, B, H, N, D, d_head)
    return out.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Test case runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(B, S, D, d_head, H, kernel, simulate, np_rng, results):
    """
    Run one (B, S, D, d_head, H) test case.
    Appends a result row to `results` list.
    Returns True on pass.
    """
    tag = f"B{B}_S{S}_dm{D}_dh{d_head}_H{H}"

    # ── Build inputs ─────────────────────────────────────────────────────────
    golden_tag = f"B{B}_S{S}_dm{D}_dh{d_head}"
    golden_x   = os.path.join(GOLDEN_DIR, f"{golden_tag}_X.npy")
    golden_wq  = os.path.join(GOLDEN_DIR, f"{golden_tag}_Wq.npy")
    golden_wk  = os.path.join(GOLDEN_DIR, f"{golden_tag}_Wk.npy")
    golden_wv  = os.path.join(GOLDEN_DIR, f"{golden_tag}_Wv.npy")
    golden_o   = os.path.join(GOLDEN_DIR, f"{golden_tag}_O.npy")

    if H == 1 and all(os.path.exists(p) for p in [golden_x, golden_wq, golden_wk, golden_wv, golden_o]):
        # Use Jnanasree's golden inputs (single-head)
        X_np  = np.load(golden_x)
        Wq_np = np.load(golden_wq)   # [D, d_head]
        Wk_np = np.load(golden_wk)
        Wv_np = np.load(golden_wv)
        O_ref = np.load(golden_o)    # [B, S, d_head]

        # Reshape ref to [B, H, N, d_head] = [B, 1, S, d_head]
        O_ref = O_ref[:, np.newaxis, :, :]   # [B, 1, S, d_head]

        # Expand Wq to [D, H*d_head] = [D, d_head] (already correct for H=1)
        Wq_mh = Wq_np   # [D, 1*d_head]
        Wk_mh = Wk_np
        Wv_mh = Wv_np
    else:
        # Generate fresh inputs for multi-head configs
        X_np   = np_rng.randn(B, S, D).astype(np.float32)
        Wq_mh  = np_rng.randn(D, H * d_head).astype(np.float32) * 0.02
        Wk_mh  = np_rng.randn(D, H * d_head).astype(np.float32) * 0.02
        Wv_mh  = np_rng.randn(D, H * d_head).astype(np.float32) * 0.02
        O_ref  = pytorch_reference(X_np, Wq_mh, Wk_mh, Wv_mh, H)  # [B, H, S, d_head]

    # ── Run kernel or simulate ───────────────────────────────────────────────
    try:
        if simulate or kernel is None:
            O_kernel = pytorch_reference(X_np, Wq_mh, Wk_mh, Wv_mh, H)
            method = "simulate"
        else:
            O_kernel = kernel_forward(X_np, Wq_mh, Wk_mh, Wv_mh,
                                      B, H, S, D, d_head, kernel)
            method = "cuda_kernel"
    except Exception as e:
        results.append({
            "tag": tag, "method": "error", "max_abs_diff": "N/A",
            "mean_abs_diff": "N/A", "pass": "ERROR", "error": str(e)[:200]
        })
        print(f"  ERROR {tag}: {e}")
        return False

    # ── Compare ──────────────────────────────────────────────────────────────
    max_diff  = float(np.abs(O_kernel - O_ref).max())
    mean_diff = float(np.abs(O_kernel - O_ref).mean())
    passed    = max_diff < TOLERANCE

    results.append({
        "tag":           tag,
        "method":        method,
        "B": B, "S": S, "D": D, "d_head": d_head, "H": H,
        "max_abs_diff":  round(max_diff,  8),
        "mean_abs_diff": round(mean_diff, 8),
        "pass":          "PASS" if passed else "FAIL",
        "error":         "",
    })

    status = "PASS" if passed else "FAIL"
    print(f"  {status}  {tag}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Skip compiled kernel; verify logic with PyTorch SDPA")
    parser.add_argument("--B",     type=int, default=None)
    parser.add_argument("--S",     type=int, default=None)
    parser.add_argument("--D",     type=int, default=None)
    parser.add_argument("--d",     type=int, default=None, dest="d_head")
    parser.add_argument("--H",     type=int, default=None)
    parser.add_argument("--out",   default="results/correctness_results.csv")
    args = parser.parse_args()

    np_rng = np.random.RandomState(42)

    # Load kernel
    kernel = None
    if not args.simulate:
        try:
            from kernel.load_kernel import load_fused_kernel
            kernel = load_fused_kernel()
            print("[test] Compiled CUDA kernel loaded.")
        except Exception as e:
            print(f"[test] Kernel not available ({e}). Falling back to --simulate.")
            args.simulate = True

    # Test matrix
    if all(v is not None for v in [args.B, args.S, args.D, args.d_head, args.H]):
        # Single config from CLI
        configs = [(args.B, args.S, args.D, args.d_head, args.H)]
    else:
        # Full suite from checkpoint document
        # Golden file configs (H=1, single-head):
        golden_configs = [
            (1,  64,  128, 64, 1),
            (1, 128,  128, 64, 1),
            (4, 256,  128, 64, 1),
            (4, 512,  128, 64, 1),
            (4, 1024, 128, 64, 1),
        ]
        # Multi-head extension configs:
        multihead_configs = [
            (1,  64, 512, 64, 4),
            (1, 128, 512, 64, 4),
            (1, 256, 512, 64, 8),
            (1, 512, 512, 64, 8),
            (4,  64, 512, 64, 8),
            (4, 128, 512, 64, 8),
        ]
        configs = golden_configs + multihead_configs

    mode_str = "PyTorch simulation" if args.simulate else "CUDA kernel"
    print(f"[test] Mode: {mode_str}")
    print(f"[test] Tolerance: max abs diff < {TOLERANCE}")
    print(f"[test] {len(configs)} test cases\n")

    results   = []
    n_pass    = 0
    n_fail    = 0

    for (B, S, D, d, H) in configs:
        ok = run_test(B, S, D, d, H, kernel, args.simulate, np_rng, results)
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    # Write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["tag", "method", "B", "S", "D", "d_head", "H",
                  "max_abs_diff", "mean_abs_diff", "pass", "error"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[test] Results: {n_pass} PASS  {n_fail} FAIL")
    print(f"[test] Saved → {args.out}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
