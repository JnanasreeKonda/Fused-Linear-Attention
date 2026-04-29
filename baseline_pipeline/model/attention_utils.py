"""
model/attention_utils.py — Shared helpers for Phase 3 attention swapping.

Provides:
  - attention-class resolution for PatchTST construction
  - checkpoint format detection
  - baseline <-> fused attention state-dict conversion
"""

from __future__ import annotations

from collections import OrderedDict


def normalize_attention_name(attention: str) -> str:
    attention = (attention or "standard").strip().lower()
    if attention not in {"standard", "fused"}:
        raise ValueError(f"Unsupported attention type: {attention}")
    return attention


def resolve_attention_block(attention: str):
    attention = normalize_attention_name(attention)
    if attention == "standard":
        return None

    from model.fused_attn_block import FusedLinearAttentionBlock

    return FusedLinearAttentionBlock


def infer_attention_from_state_dict(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any(".attn.Wq" in key for key in keys):
        return "fused"
    if any(".attn.q_proj.weight" in key for key in keys):
        return "standard"
    return "unknown"


def convert_state_dict_for_attention(
    state_dict: dict,
    source_attention: str,
    target_attention: str,
) -> OrderedDict:
    """
    Convert checkpoint weights between PatchTST attention implementations.

    StandardAttentionBlock stores projection weights in nn.Linear format:
      weight shape = [out_dim, in_dim]

    FusedLinearAttentionBlock stores projection matrices in matmul format:
      Wq / Wk / Wv shape = [in_dim, out_dim]

    The conversion is therefore a transpose for Q/K/V only. The output
    projection remains an nn.Linear in both implementations.
    """

    source_attention = normalize_attention_name(source_attention)
    target_attention = normalize_attention_name(target_attention)

    if source_attention == target_attention:
        return OrderedDict((key, value) for key, value in state_dict.items())

    converted = OrderedDict()
    for key, value in state_dict.items():
        if source_attention == "standard" and target_attention == "fused":
            if key.endswith(".attn.q_proj.weight"):
                converted[key.replace(".attn.q_proj.weight", ".attn.Wq")] = (
                    value.t().contiguous()
                )
                continue
            if key.endswith(".attn.k_proj.weight"):
                converted[key.replace(".attn.k_proj.weight", ".attn.Wk")] = (
                    value.t().contiguous()
                )
                continue
            if key.endswith(".attn.v_proj.weight"):
                converted[key.replace(".attn.v_proj.weight", ".attn.Wv")] = (
                    value.t().contiguous()
                )
                continue

        if source_attention == "fused" and target_attention == "standard":
            if key.endswith(".attn.Wq"):
                converted[key.replace(".attn.Wq", ".attn.q_proj.weight")] = (
                    value.t().contiguous()
                )
                continue
            if key.endswith(".attn.Wk"):
                converted[key.replace(".attn.Wk", ".attn.k_proj.weight")] = (
                    value.t().contiguous()
                )
                continue
            if key.endswith(".attn.Wv"):
                converted[key.replace(".attn.Wv", ".attn.v_proj.weight")] = (
                    value.t().contiguous()
                )
                continue

        converted[key] = value

    return converted
