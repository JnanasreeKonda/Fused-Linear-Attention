"""
tests/test_fused_backward.py — Smoke-test the custom fused backward bridge.

This script does not require CUDA. It injects a fake kernel forward that
matches the PyTorch reference attention math, then verifies that the custom
autograd bridge in `model/fused_attn_block.py` returns the same gradients as
native PyTorch autograd.
"""

from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_pipeline.model.fused_attn_block import (  # noqa: E402
    _FusedAttentionAutogradFn,
    _reference_attention_forward,
)


class _FakeKernel:
    def forward(self, x, q_w, k_w, v_w, B, H, S, D, d_head):
        out = _reference_attention_forward(
            x,
            q_w,
            k_w,
            v_w,
            n_heads=H,
            d_head=d_head,
            dropout_p=0.0,
            training=False,
        )
        return out.view(B, S, H, d_head).transpose(1, 2).contiguous()


def clone_with_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def main():
    torch.manual_seed(7)

    B, S, D, H = 2, 5, 8, 2
    d_head = D // H

    x = torch.randn(B, S, D, dtype=torch.float32)
    q_w = torch.randn(D, D, dtype=torch.float32) * 0.05
    k_w = torch.randn(D, D, dtype=torch.float32) * 0.05
    v_w = torch.randn(D, D, dtype=torch.float32) * 0.05
    grad_out = torch.randn(B, S, D, dtype=torch.float32)

    x_ref = clone_with_grad(x)
    q_ref = clone_with_grad(q_w)
    k_ref = clone_with_grad(k_w)
    v_ref = clone_with_grad(v_w)
    y_ref = _reference_attention_forward(
        x_ref,
        q_ref,
        k_ref,
        v_ref,
        n_heads=H,
        d_head=d_head,
        dropout_p=0.0,
        training=False,
    )
    y_ref.backward(grad_out)

    x_fused = clone_with_grad(x)
    q_fused = clone_with_grad(q_w)
    k_fused = clone_with_grad(k_w)
    v_fused = clone_with_grad(v_w)
    y_fused = _FusedAttentionAutogradFn.apply(
        x_fused,
        q_fused,
        k_fused,
        v_fused,
        H,
        d_head,
        _FakeKernel(),
    )
    y_fused.backward(grad_out)

    checks = {
        "x": (x_ref.grad, x_fused.grad),
        "Wq": (q_ref.grad, q_fused.grad),
        "Wk": (k_ref.grad, k_fused.grad),
        "Wv": (v_ref.grad, v_fused.grad),
    }

    tol = 1e-5
    for name, (grad_expected, grad_actual) in checks.items():
        max_diff = (grad_expected - grad_actual).abs().max().item()
        print(f"[backward] {name}: max_diff={max_diff:.3e}")
        if max_diff > tol:
            raise SystemExit(
                f"Backward mismatch for {name}: max_diff={max_diff:.3e} > tol={tol:.1e}"
            )

    print("[backward] PASS")


if __name__ == "__main__":
    main()
