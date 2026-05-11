"""
model/fused_attn_block.py — Fused attention block wrapper.
Owner: Rithwik Amajala (integration) + Jnanasree (kernel)  |  M10  |  Phase 3

This module wraps the compiled CUDA kernel as a drop-in StandardAttentionBlock
replacement. The forward path uses Jnanasree's fused CUDA kernel; the backward
path recomputes the reference attention graph under autograd so end-to-end
training can stay on the fused forward path without requiring a second custom
CUDA backward kernel.
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

DEFAULT_TILE_SIZE = int(os.environ.get("FLA_TILE_SIZE", "64"))
DEFAULT_KERNEL_DTYPE = os.environ.get("FLA_KERNEL_DTYPE", "float32").lower()
DEFAULT_ATTN_BACKEND = os.environ.get("FLA_ATTN_BACKEND", "fused").lower()


def _kernel_uses_low_precision() -> bool:
    return DEFAULT_KERNEL_DTYPE in {"float16", "fp16", "f16", "half", "bfloat16", "bf16"}


def _reference_attention_forward(
    x: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    n_heads: int,
    d_head: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    B, S, D = x.shape
    q = (x @ q_w).view(B, S, n_heads, d_head).transpose(1, 2)
    k = (x @ k_w).view(B, S, n_heads, d_head).transpose(1, 2)
    v = (x @ v_w).view(B, S, n_heads, d_head).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=(dropout_p if training else 0.0),
    )
    return out.transpose(1, 2).contiguous().view(B, S, D)


class _FusedAttentionAutogradFn(torch.autograd.Function):
    """
    CUDA-forward / PyTorch-backward bridge for the fused attention block.

    Forward:
      - uses the compiled custom CUDA kernel
    Backward:
      - recomputes the unfused PyTorch attention graph and differentiates it
        with autograd to obtain exact gradients for X/Wq/Wk/Wv

    This keeps the assignment deliverable practical: the custom fused kernel is
    used for the fast forward path, and training remains end-to-end usable on
    CUDA without writing a second, much larger, handwritten CUDA backward.
    """

    @staticmethod
    def forward(ctx, x, q_w, k_w, v_w, n_heads, d_head, kernel):
        B, S, D = x.shape
        out = kernel.forward(
            x.contiguous(),
            q_w.contiguous(),
            k_w.contiguous(),
            v_w.contiguous(),
            B,
            n_heads,
            S,
            D,
            d_head,
        )
        ctx.save_for_backward(x, q_w, k_w, v_w)
        ctx.n_heads = int(n_heads)
        ctx.d_head = int(d_head)
        return out.transpose(1, 2).contiguous().view(B, S, D)

    @staticmethod
    def backward(ctx, grad_out):
        x, q_w, k_w, v_w = ctx.saved_tensors
        needs = ctx.needs_input_grad[:4]

        with torch.enable_grad():
            x_ref = x.detach().requires_grad_(needs[0])
            q_w_ref = q_w.detach().requires_grad_(needs[1])
            k_w_ref = k_w.detach().requires_grad_(needs[2])
            v_w_ref = v_w.detach().requires_grad_(needs[3])

            out_ref = _reference_attention_forward(
                x_ref,
                q_w_ref,
                k_w_ref,
                v_w_ref,
                n_heads=ctx.n_heads,
                d_head=ctx.d_head,
                dropout_p=0.0,
                training=False,
            )

            grads = torch.autograd.grad(
                outputs=out_ref,
                inputs=(x_ref, q_w_ref, k_w_ref, v_w_ref),
                grad_outputs=grad_out,
                allow_unused=True,
            )

        return grads[0], grads[1], grads[2], grads[3], None, None, None


class _HybridAttentionAutogradFn(torch.autograd.Function):
    """
    Hybrid path:
      - projections use PyTorch matmul
      - custom CUDA kernel computes only the attention stage
      - backward recomputes the full reference attention graph
    """

    @staticmethod
    def forward(ctx, x, q_w, k_w, v_w, n_heads, d_head, kernel):
        B, S, D = x.shape
        q = (x @ q_w).view(B, S, n_heads, d_head).transpose(1, 2).contiguous()
        k = (x @ k_w).view(B, S, n_heads, d_head).transpose(1, 2).contiguous()
        v = (x @ v_w).view(B, S, n_heads, d_head).transpose(1, 2).contiguous()
        out = kernel.forward(q, k, v, B, n_heads, S, d_head)
        ctx.save_for_backward(x, q_w, k_w, v_w)
        ctx.n_heads = int(n_heads)
        ctx.d_head = int(d_head)
        return out.transpose(1, 2).contiguous().view(B, S, D)

    @staticmethod
    def backward(ctx, grad_out):
        x, q_w, k_w, v_w = ctx.saved_tensors
        needs = ctx.needs_input_grad[:4]

        with torch.enable_grad():
            x_ref = x.detach().requires_grad_(needs[0])
            q_w_ref = q_w.detach().requires_grad_(needs[1])
            k_w_ref = k_w.detach().requires_grad_(needs[2])
            v_w_ref = v_w.detach().requires_grad_(needs[3])

            out_ref = _reference_attention_forward(
                x_ref,
                q_w_ref,
                k_w_ref,
                v_w_ref,
                n_heads=ctx.n_heads,
                d_head=ctx.d_head,
                dropout_p=0.0,
                training=False,
            )

            grads = torch.autograd.grad(
                outputs=out_ref,
                inputs=(x_ref, q_w_ref, k_w_ref, v_w_ref),
                grad_outputs=grad_out,
                allow_unused=True,
            )

        return grads[0], grads[1], grads[2], grads[3], None, None, None


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

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._warned_reference_fallback = False

    def _load_kernel(self):
        if self._kernel is None:
            try:
                if DEFAULT_ATTN_BACKEND == "hybrid":
                    from kernel.load_attn_only import load_attn_only_kernel

                    self._kernel = load_attn_only_kernel(
                        head_dim=self.d_head,
                        tile_size=DEFAULT_TILE_SIZE,
                        kernel_dtype=DEFAULT_KERNEL_DTYPE,
                    )
                else:
                    from kernel.load_kernel import load_fused_kernel

                    self._kernel = load_fused_kernel(
                        head_dim=self.d_head,
                        tile_size=DEFAULT_TILE_SIZE,
                        kernel_dtype=DEFAULT_KERNEL_DTYPE,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "FusedLinearAttentionBlock: kernel unavailable.\n"
                    "Falling back to the PyTorch reference path is supported,\n"
                    "but compiled fused-kernel execution requires a working CUDA\n"
                    "toolchain plus Jnanasree's M8 handoff.\n"
                    f"Original error: {exc}"
                )

    def _reference_attention(self, x: torch.Tensor) -> torch.Tensor:
        q_w = self.q_proj.weight.t().contiguous()
        k_w = self.k_proj.weight.t().contiguous()
        v_w = self.v_proj.weight.t().contiguous()
        out = _reference_attention_forward(
            x,
            q_w,
            k_w,
            v_w,
            n_heads=self.n_heads,
            d_head=self.d_head,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out_proj(out)

    def _should_use_reference_path(self, x: torch.Tensor) -> bool:
        if x.device.type != "cuda":
            return True
        if x.dtype != torch.float32:
            return True
        if self.training and _kernel_uses_low_precision():
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
        if x.dtype != torch.float32:
            reasons.append(f"unsupported dtype={x.dtype}")
        if self.training and _kernel_uses_low_precision():
            reasons.append("training with low-precision fused kernel")
        if self.training and self.dropout > 0:
            reasons.append("training-time attention dropout")

        msg = ", ".join(reasons) if reasons else "runtime constraint"
        print(
            "[fused_attn_block] Falling back to PyTorch reference attention "
            f"({msg})."
        )
        self._warned_reference_fallback = True

    def _fused_attention(self, x: torch.Tensor) -> torch.Tensor:
        if DEFAULT_KERNEL_DTYPE in {"bfloat16", "bf16"}:
            kernel_dtype = torch.bfloat16
        elif _kernel_uses_low_precision():
            kernel_dtype = torch.float16
        else:
            kernel_dtype = torch.float32
        x_kernel = x.contiguous().to(kernel_dtype)
        q_w = self.q_proj.weight.t().contiguous().to(kernel_dtype)
        k_w = self.k_proj.weight.t().contiguous().to(kernel_dtype)
        v_w = self.v_proj.weight.t().contiguous().to(kernel_dtype)
        if DEFAULT_ATTN_BACKEND == "hybrid":
            out = _HybridAttentionAutogradFn.apply(
                x_kernel,
                q_w,
                k_w,
                v_w,
                self.n_heads,
                self.d_head,
                self._kernel,
            )
        else:
            out = _FusedAttentionAutogradFn.apply(
                x_kernel,
                q_w,
                k_w,
                v_w,
                self.n_heads,
                self.d_head,
                self._kernel,
            )
        return self.out_proj(out)

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

        return self._fused_attention(x)
