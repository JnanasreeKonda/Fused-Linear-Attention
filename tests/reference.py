"""
tests/reference.py — Canonical NumPy CPU reference implementation.

This mirrors the implemented oracle in `CPU_Reference_in_NumPy/tests/reference.py`
so the root `tests/` package is self-contained.
"""

from __future__ import annotations

import numpy as np


def split_qkv_weight(W_qkv: np.ndarray):
    """
    Split a stacked fused-projection weight matrix into Q/K/V components.

    Parameters
    ----------
    W_qkv:
        Array with shape (3 * d_model, d_model), matching the PyTorch
        `F.linear(x, weight)` layout for a stacked QKV projection.
    """
    W_qkv = np.asarray(W_qkv, dtype=np.float32)
    if W_qkv.ndim != 2:
        raise ValueError(f"W_qkv must be 2-D, got shape {W_qkv.shape}")
    if W_qkv.shape[0] % 3 != 0:
        raise ValueError(f"First dimension of W_qkv must be divisible by 3, got {W_qkv.shape}")

    d_model = W_qkv.shape[1]
    qkv_out_dim = W_qkv.shape[0] // 3
    return np.split(W_qkv, 3, axis=0), d_model, qkv_out_dim


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fused_qkv_attention_reference(
    X,
    W_q,
    W_k,
    W_v,
):
    """
    CPU NumPy reference: QKV projection fused with scaled dot-product attention.

    Returns
    -------
    tuple
        (O, Q, K, V, attn_weights)
    """
    d_head = W_q.shape[1]
    scale = 1.0 / np.sqrt(d_head)

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = (Q @ K.transpose(0, 2, 1)) * scale
    attn_weights = softmax(scores, axis=-1)
    O = attn_weights @ V

    return O, Q, K, V, attn_weights


def fused_attention_reference(X, W_qkv, n_heads):
    """
    Canonical root-level oracle for fused projection + attention.

    Parameters
    ----------
    X:
        Input activations with shape (B, S, D).
    W_qkv:
        Stacked QKV projection weight with shape (3D, D).
    n_heads:
        Number of attention heads. The current NumPy oracle uses this for input
        validation and to mirror the model-side interface even though the
        single-matmul formulation returns a merged `(B, S, D)` output.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (B, S, D), got {X.shape}")

    batch, seq_len, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    d_head = d_model // n_heads
    (W_q, W_k, W_v), in_dim, qkv_out_dim = split_qkv_weight(W_qkv)
    if in_dim != d_model:
        raise ValueError(f"W_qkv input dim {in_dim} must match X feature dim {d_model}")
    if qkv_out_dim % n_heads != 0:
        raise ValueError(
            f"Stacked QKV output dim {qkv_out_dim} must be divisible by n_heads={n_heads}"
        )

    qkv_d_head = qkv_out_dim // n_heads
    Q = (X @ W_q.T).reshape(batch, seq_len, n_heads, qkv_d_head).transpose(0, 2, 1, 3)
    K = (X @ W_k.T).reshape(batch, seq_len, n_heads, qkv_d_head).transpose(0, 2, 1, 3)
    V = (X @ W_v.T).reshape(batch, seq_len, n_heads, qkv_d_head).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(qkv_d_head))
    attn_weights = softmax(scores, axis=-1).astype(np.float32, copy=False)
    out = attn_weights @ V
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, qkv_out_dim)
    return out.astype(np.float32, copy=False)


def run_reference_checks():
    np.random.seed(42)

    configs = [
        (1, 64, 128, 64, 1),
        (1, 128, 128, 64, 1),
        (4, 256, 128, 64, 1),
        (4, 512, 128, 64, 1),
        (2, 64, 128, 64, 2),
    ]

    for (B, S, d_model, d_head, n_heads) in configs:
        X = np.random.randn(B, S, d_model).astype(np.float32)
        W_q = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
        W_k = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
        W_v = np.random.randn(d_model, d_head).astype(np.float32) * 0.02

        O, _, _, _, weights = fused_qkv_attention_reference(X, W_q, W_k, W_v)
        assert O.shape == (B, S, d_head), f"Wrong output shape: {O.shape}"
        assert not np.any(np.isnan(O)), "NaN in output"
        assert not np.any(np.isinf(O)), "Inf in output"

        if n_heads == 1:
            W_qkv = np.concatenate([W_q.T, W_k.T, W_v.T], axis=0)
            fused_out = fused_attention_reference(X, W_qkv, n_heads=1)
            assert np.allclose(O, fused_out, atol=1e-5), "Stacked oracle must match split reference"
        else:
            W_q_full = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
            W_k_full = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
            W_v_full = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
            W_qkv = np.concatenate([W_q_full, W_k_full, W_v_full], axis=0)
            fused_out = fused_attention_reference(X, W_qkv, n_heads=n_heads)
            expected_shape = (B, S, d_model)
            assert fused_out.shape == expected_shape, f"Wrong stacked output shape: {fused_out.shape}"
            assert not np.any(np.isnan(fused_out)), "NaN in stacked oracle output"
            assert not np.any(np.isinf(fused_out)), "Inf in stacked oracle output"

        weight_sums = weights.sum(axis=-1)
        assert np.allclose(weight_sums, 1.0, atol=1e-5), (
            f"Attention weights don't sum to 1: max err {np.abs(weight_sums - 1).max()}"
        )

        print(
            f"PASS B={B} S={S:4d} d_model={d_model} d_head={d_head} "
            f"| O: {O.shape} | weights sum err: {np.abs(weight_sums - 1).max():.2e}"
        )

    print("\nAll reference checks passed.")


if __name__ == "__main__":
    run_reference_checks()
