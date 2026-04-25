"""
tests/reference.py — Canonical NumPy CPU reference implementation.

This mirrors the implemented oracle in `CPU_Reference_in_NumPy/tests/reference.py`
so the root `tests/` package is self-contained.
"""

from __future__ import annotations

import numpy as np


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


def run_reference_checks():
    np.random.seed(42)

    configs = [
        (1, 64, 128, 64),
        (1, 128, 128, 64),
        (4, 256, 128, 64),
        (4, 512, 128, 64),
        (4, 1024, 128, 64),
    ]

    for (B, S, d_model, d_head) in configs:
        X = np.random.randn(B, S, d_model).astype(np.float32)
        W_q = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
        W_k = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
        W_v = np.random.randn(d_model, d_head).astype(np.float32) * 0.02

        O, _, _, _, weights = fused_qkv_attention_reference(X, W_q, W_k, W_v)

        assert O.shape == (B, S, d_head), f"Wrong output shape: {O.shape}"
        assert not np.any(np.isnan(O)), "NaN in output"
        assert not np.any(np.isinf(O)), "Inf in output"

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
