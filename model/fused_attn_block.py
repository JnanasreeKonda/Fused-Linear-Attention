"""
model/fused_attn_block.py — Fused attention block wrapper.
Owner: Rithwik Amajala (integration) + Jnanasree (kernel)  |  M10  |  Phase 3

PLACEHOLDER — do not fill until Jnanasree delivers load_kernel.py (end of M8).

This module wraps the compiled CUDA kernel as a drop-in StandardAttentionBlock
replacement.  Once Jnanasree hands off M8, Rithwik swaps this into PatchTST
by passing:

    model = PatchTST(attn_block_class=FusedLinearAttentionBlock)

and retrains from scratch with the same seed / hyperparameters.
"""

import torch
import torch.nn as nn


class FusedLinearAttentionBlock(nn.Module):
    """
    Drop-in replacement for StandardAttentionBlock using the custom CUDA kernel.

    Interface (must match StandardAttentionBlock exactly):
      __init__(d_model: int, n_heads: int, dropout: float)
      forward(x: Tensor[B, S, D]) -> Tensor[B, S, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.dropout  = dropout
        self._kernel  = None   # set in _load_kernel()

        # Weight tensors — kept as Parameters so optimiser can update them.
        # Shape mirrors the fused weight matrix W ∈ R^{3D × D} expected by
        # the CUDA kernel (Q, K, V stacked row-wise).
        self.qkv_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.qkv_weight)

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
        self._load_kernel()
        # TODO (Phase 3): call self._kernel.fused_linear_attention(x, self.qkv_weight, ...)
        raise NotImplementedError(
            "FusedLinearAttentionBlock.forward() will be implemented "
            "during Phase 3 (M10) once the CUDA kernel is available."
        )
