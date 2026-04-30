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
    if attention not in {"standard", "fused", "fused_legacy"}:
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
        return "fused_legacy"
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

    Current StandardAttentionBlock and FusedLinearAttentionBlock share the same
    q_proj / k_proj / v_proj / out_proj state-dict layout. Conversion is only
    needed for legacy fused checkpoints that stored Wq/Wk/Wv matrices directly.
    """

    source_attention = normalize_attention_name(source_attention)
    target_attention = normalize_attention_name(target_attention)

    if source_attention != "fused_legacy" and target_attention != "fused_legacy":
        return OrderedDict((key, value) for key, value in state_dict.items())

    converted = OrderedDict()
    for key, value in state_dict.items():
        if source_attention == "fused_legacy" and target_attention in {"standard", "fused"}:
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

        if source_attention in {"standard", "fused"} and target_attention == "fused_legacy":
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

        converted[key] = value

    return converted
