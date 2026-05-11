"""
kernel/load_attn_only_warp4.py — JIT loader for the warp-cooperative attention-only kernel.
"""

from __future__ import annotations

import os
from typing import Optional

_kernel_cache = {}


def _default_cuda_arch() -> str:
    override = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:
        pass
    return "8.0"


def _normalize_kernel_dtype(kernel_dtype: Optional[str]) -> str:
    value = kernel_dtype or os.environ.get("FLA_KERNEL_DTYPE", "float32")
    normalized = value.lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "f32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "f16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported kernel dtype '{value}'")
    return aliases[normalized]


def load_attn_only_warp4_kernel(
    head_dim: int = 64,
    tile_size: int = 64,
    kernel_dtype: Optional[str] = None,
    q_group_size: Optional[int] = None,
):
    global _kernel_cache
    normalized_dtype = _normalize_kernel_dtype(kernel_dtype)
    q_group = int(q_group_size or os.environ.get("FLA_Q_GROUP_SIZE", "2"))
    cache_key = (int(head_dim), int(tile_size), normalized_dtype, q_group)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file = os.path.join(root, "kernel", "attn_only_warp4.cu")
    cpp_file = os.path.join(root, "kernel", "attn_only_warp4_ext.cpp")
    build_dir = os.path.join(root, "build", "attn_only_warp4")
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _default_cuda_arch())

    dtype_tag = "f32"
    if normalized_dtype == "float16":
        dtype_tag = "f16"
    elif normalized_dtype == "bfloat16":
        dtype_tag = "bf16"
    module_name = (
        f"attn_only_warp4_hd{head_dim}_tile{tile_size}_{dtype_tag}_qg{q_group}"
    )

    _kernel_cache[cache_key] = load(
        name=module_name,
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-DTILE_SIZE={int(tile_size)}",
            f"-DHEAD_DIM={int(head_dim)}",
            f"-DQ_GROUP_SIZE={q_group}",
            "-allow-unsupported-compiler",
        ],
        verbose=False,
        build_directory=build_dir,
    )
    return _kernel_cache[cache_key]
