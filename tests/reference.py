"""
tests/reference.py — NumPy CPU reference implementation.
Owner: Jnanasree  |  Milestone: M4  |  Phase 1

PLACEHOLDER — Jnanasree will implement this.
This module provides the mathematical ground-truth oracle that all GPU
kernel correctness tests compare against (max abs diff < 1e-4).
"""

import numpy as np

# TODO (Jnanasree, M4): implement fused_attention_reference(X, W_qkv, n_heads)
# Should compute:
#   Q, K, V = X @ W_Q.T,  X @ W_K.T,  X @ W_V.T
#   scores  = Q @ K.T / sqrt(d_head)
#   weights = softmax(scores, axis=-1)
#   output  = weights @ V
# entirely in NumPy; returns np.float32 array of shape (B, S, D)

def fused_attention_reference(X: np.ndarray, W_qkv: np.ndarray, n_heads: int) -> np.ndarray:
    raise NotImplementedError(
        "reference.py is a Phase 1 deliverable (M4 — Jnanasree).\n"
        "It provides the NumPy oracle for GPU kernel correctness testing."
    )
