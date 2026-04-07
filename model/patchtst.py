"""
model/patchtst.py — Lightweight PatchTST with swappable AttentionBlock.
Owner: Rithwik Amajala  |  Milestone: M5  |  Phase 1

Architecture (channel-independent PatchTST):
  Input  : (B, input_len, n_vars)
  ↓ unfold patches
  Patches: (B * n_vars, n_patches, patch_len)
  ↓ linear patch embedding
  Tokens : (B * n_vars, n_patches, d_model)
  ↓ N_LAYERS × PatchTSTLayer  [AttentionBlock + FFN + LayerNorm]
  ↓ head
  Output : (B, forecast_len)   — mean across variables

The `attn_block_class` argument is the drop-in swap point.
Phase 1 uses StandardAttentionBlock (unfused Q/K/V + SDPA).
Phase 3 will pass FusedLinearAttentionBlock here.

Usage (smoke-test):
    python model/patchtst.py
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ──────────────────────────────────────────────────────────────────────────────
# Attention Block Interface
# ──────────────────────────────────────────────────────────────────────────────

class StandardAttentionBlock(nn.Module):
    """
    Baseline (unfused) attention block: 3 separate Linear projections + SDPA.

    This is the two-kernel pipeline that FusedLinearAttention replaces:
      Kernel 1 — QKV matmuls  (writes QKV to HBM)
      Kernel 2 — SDPA         (reads QKV from HBM)

    Interface contract (all attention blocks must honour this):
      __init__(d_model: int, n_heads: int, dropout: float)
      forward(x: Tensor[B, S, D]) -> Tensor[B, S, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        # Three separate Linear layers ← unfused projection (baseline)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # ── Kernel 1: QKV projection ──────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # ── Kernel 2: Scaled Dot-Product Attention ────────────────────────────
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learnt)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Layer
# ──────────────────────────────────────────────────────────────────────────────

class PatchTSTLayer(nn.Module):
    """Pre-norm transformer encoder layer with swappable attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_block_class=None,
    ):
        super().__init__()
        if attn_block_class is None:
            attn_block_class = StandardAttentionBlock

        self.attn  = attn_block_class(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# PatchTST Model
# ──────────────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Lightweight PatchTST for ETTh1 forecasting.

    Parameters
    ----------
    n_vars           : number of input features (7 for ETTh1)
    input_len        : lookback window (time steps)
    forecast_len     : forecast horizon (time steps)
    patch_len        : patch size in time steps
    stride           : patch stride
    d_model          : token / embed dimension
    n_heads          : attention heads per layer
    n_layers         : number of transformer encoder layers
    d_ff             : FFN hidden dimension
    dropout          : dropout rate
    attn_block_class : class for the attention sub-layer (swap here for fused kernel)
                       Must accept __init__(d_model, n_heads, dropout)
                       and forward(x: [B,S,D]) -> [B,S,D]
    """

    def __init__(
        self,
        n_vars: int          = len(["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]),
        input_len: int       = config.INPUT_LEN,
        forecast_len: int    = config.FORECAST_LEN,
        patch_len: int       = config.PATCH_LEN,
        stride: int          = config.STRIDE,
        d_model: int         = config.D_MODEL,
        n_heads: int         = config.N_HEADS,
        n_layers: int        = config.N_LAYERS,
        d_ff: int            = config.D_FF,
        dropout: float       = config.DROPOUT,
        attn_block_class     = None,
    ):
        super().__init__()
        self.n_vars       = n_vars
        self.input_len    = input_len
        self.forecast_len = forecast_len
        self.patch_len    = patch_len
        self.stride       = stride

        # n_patches per channel: (input_len - patch_len) // stride + 1
        # e.g. (96 - 16) // 8 + 1 = 11
        self.n_patches = (input_len - patch_len) // stride + 1

        # Patch embedding: map raw patch values → d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Positional encoding (sized to n_patches)
        self.pos_enc = PositionalEncoding(
            d_model, max_len=self.n_patches + 8, dropout=dropout
        )

        # Transformer encoder (N_LAYERS layers, all sharing the same attn class)
        self.encoder = nn.ModuleList(
            [
                PatchTSTLayer(d_model, n_heads, d_ff, dropout, attn_block_class)
                for _ in range(n_layers)
            ]
        )

        # Forecasting head: flatten patch representations → forecast_len
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, forecast_len)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, input_len, n_vars)

        Returns
        -------
        (B, forecast_len) — normalised OT forecast
        """
        B, T, C = x.shape

        # ── Channel-independent patching ──────────────────────────────────────
        # (B, T, C) → (B, C, T) → unfold → (B, C, n_patches, patch_len)
        x = x.permute(0, 2, 1)                                          # (B, C, T)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, C, n_patches, patch_len)
        B2, C2, N, L = x.shape   # N == self.n_patches, L == self.patch_len
        tokens = x.reshape(B2 * C2, N, L)                               # (B*C, n_patches, patch_len)

        # ── Patch embedding + positional encoding ─────────────────────────────
        tokens = self.patch_embed(tokens)                  # (B*C, n_patches, d_model)
        tokens = self.pos_enc(tokens)

        # ── Transformer encoder ───────────────────────────────────────────────
        for layer in self.encoder:
            tokens = layer(tokens)                         # (B*C, n_patches, d_model)

        # ── Forecasting head ──────────────────────────────────────────────────
        tokens = self.head_norm(tokens)
        tokens = tokens.reshape(B2 * C2, -1)               # (B*C, n_patches*d_model)
        forecast = self.head(tokens)                        # (B*C, forecast_len)

        # ── Average across channels to produce OT forecast ───────────────────
        forecast = forecast.view(B2, C2, self.forecast_len).mean(dim=1)  # (B, forecast_len)
        return forecast


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = PatchTST()
    dummy = torch.randn(4, config.INPUT_LEN, 7)
    out   = model(dummy)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[patchtst] Input  : {tuple(dummy.shape)}")
    print(f"[patchtst] Output : {tuple(out.shape)}")
    print(f"[patchtst] Params : {n_params:,}")
    print(f"[patchtst] n_patches = {model.n_patches}")
    assert out.shape == (4, config.FORECAST_LEN), f"Shape mismatch: {out.shape}"
    print("[patchtst] ✓ patchtst.py smoke-test passed.")
