"""
model/fused_attn_block.py — Fused attention block wrapper.
Owner: Rithwik Amajala (integration) + Jnanasree (kernel)  |  M10  |  Phase 3

This module wraps the compiled CUDA kernel as a drop-in StandardAttentionBlock
replacement.  Once Jnanasree hands off M8, Rithwik swaps this into PatchTST
by passing:

    model = PatchTST(attn_block_class=FusedLinearAttentionBlock)

and retrains from scratch with the same seed / hyperparameters.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class FusedLinearAttentionBlock(nn.Module):
    """
    Drop-in replacement for StandardAttentionBlock using the custom CUDA kernel.

    Interface (must match StandardAttentionBlock exactly):
      __init__(d_model: int, n_heads: int, dropout: float)
      forward(x: Tensor[B, S, D]) -> Tensor[B, S, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.dropout  = dropout
        self._kernel  = None   # set in _load_kernel()

        # The canonical kernel API expects separate Q/K/V matrices of shape
        # [D, H * d_head], matching the unfused block's projection layout.
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)

    def _load_kernel(self):
        """JIT-compile and cache the CUDA extension (first call only)."""
        if self._kernel is None:
            try:
                from kernel.load_kernel import load_fused_kernel
                self._kernel = load_fused_kernel()
            except ImportError as exc:
                raise RuntimeError(
                    "FusedLinearAttentionBlock: kernel not available yet.\n"
                    "Wait for Jnanasree's M8 handoff (load_kernel.py).\n"
                    f"Original error: {exc}"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.d_head != 64:
            raise RuntimeError(
                "The current fused kernel is compiled for d_head=64 only. "
                f"Received d_head={self.d_head}. "
                "Either generalize kernel/load_kernel.py for multiple head "
                "dimensions or use the benchmark configuration for fused runs."
            )
        if x.device.type != "cuda":
            raise RuntimeError("FusedLinearAttentionBlock requires CUDA input tensors.")

        self._load_kernel()
        B, S, D = x.shape
        out = self._kernel.forward(
            x.contiguous(),
            self.Wq.contiguous(),
            self.Wk.contiguous(),
            self.Wv.contiguous(),
            B,
            self.n_heads,
            S,
            D,
            self.d_head,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)
