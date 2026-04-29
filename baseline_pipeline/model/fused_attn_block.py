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
import torch.nn.functional as F

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

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self._kernel = None

        # The canonical kernel API expects separate Q/K/V matrices of shape
        # [D, H * d_head], matching X @ W projection layout.
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self._warned_reference_fallback = False

    def _load_kernel(self):
        """JIT-compile and cache the CUDA extension (first call only)."""
        if self._kernel is None:
            try:
                from kernel.load_kernel import load_fused_kernel

                self._kernel = load_fused_kernel(head_dim=self.d_head)
            except Exception as exc:
                raise RuntimeError(
                    "FusedLinearAttentionBlock: kernel unavailable.\n"
                    "Falling back to the PyTorch reference path is supported,\n"
                    "but compiled fused-kernel inference requires a working CUDA\n"
                    "toolchain plus Jnanasree's M8 handoff.\n"
                    f"Original error: {exc}"
                )

    def _reference_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = torch.matmul(x, self.Wq).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = torch.matmul(x, self.Wk).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = torch.matmul(x, self.Wv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

    def _should_use_reference_path(self, x: torch.Tensor) -> bool:
        if x.device.type != "cuda":
            return True
        if torch.is_grad_enabled():
            return True
        if self.training and self.dropout > 0:
            return True
        return False

    def _warn_reference_fallback(self, x: torch.Tensor) -> None:
        if self._warned_reference_fallback:
            return

        reasons = []
        if x.device.type != "cuda":
            reasons.append("non-CUDA device")
        if torch.is_grad_enabled():
            reasons.append("autograd-enabled execution")
        if self.training and self.dropout > 0:
            reasons.append("training-time dropout")

        msg = ", ".join(reasons) if reasons else "runtime constraint"
        print(
            "[fused_attn_block] Falling back to PyTorch reference attention "
            f"({msg})."
        )
        self._warned_reference_fallback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_use_reference_path(x):
            self._warn_reference_fallback(x)
            return self._reference_attention(x)

        try:
            self._load_kernel()
        except RuntimeError as exc:
            if not self._warned_reference_fallback:
                print(f"[fused_attn_block] {exc}")
            self._warn_reference_fallback(x)
            return self._reference_attention(x)

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
