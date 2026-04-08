# tests/test_reference_vs_pytorch.py
import numpy as np
import torch
import torch.nn.functional as F
from reference import fused_qkv_attention_reference

def test_against_pytorch(B=2, S=64, d_model=128, d_head=64, tol=1e-4):
    np.random.seed(0)
    torch.manual_seed(0)

    X_np   = np.random.randn(B, S, d_model).astype(np.float32)
    W_q_np = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
    W_k_np = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
    W_v_np = np.random.randn(d_model, d_head).astype(np.float32) * 0.02

    # --- NumPy reference ---
    O_ref, Q_ref, K_ref, V_ref, _ = fused_qkv_attention_reference(
        X_np, W_q_np, W_k_np, W_v_np
    )

    # --- PyTorch reference ---
    X_t   = torch.from_numpy(X_np)
    W_q_t = torch.from_numpy(W_q_np)
    W_k_t = torch.from_numpy(W_k_np)
    W_v_t = torch.from_numpy(W_v_np)

    Q_t = X_t @ W_q_t
    K_t = X_t @ W_k_t
    V_t = X_t @ W_v_t

    # Use PyTorch's SDPA (same math, just verifying)
    scale = 1.0 / (d_head ** 0.5)
    scores_t = (Q_t @ K_t.transpose(-2, -1)) * scale
    O_pt = (torch.softmax(scores_t, dim=-1) @ V_t).numpy()

    max_diff = np.abs(O_ref - O_pt).max()
    print(f"Max absolute diff (NumPy ref vs PyTorch): {max_diff:.2e}")
    assert max_diff < tol, f"FAIL: diff {max_diff} exceeds tolerance {tol}"
    print(f"PASS: NumPy reference matches PyTorch (tol={tol})")

if __name__ == "__main__":
    test_against_pytorch()