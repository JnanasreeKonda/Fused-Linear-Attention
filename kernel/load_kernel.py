"""
kernel/load_kernel.py — JIT-compile and load the canonical FusedLinearAttention extension.

This is the single source of truth for loading the CUDA kernel from the root
`kernel/` package. Scripts inside `baseline_pipeline/` should import this file
instead of reaching into any archived bundle directory.
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


def _default_tile_size() -> int:
    override = os.environ.get("FLA_TILE_SIZE")
    if override:
        return int(override)

    try:
        import torch

        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major >= 9:
                return 64
    except Exception:
        pass

    return 32


def _default_proj_k_tile(kernel_dtype: Optional[str] = None) -> int:
    override = os.environ.get("FLA_PROJ_K_TILE")
    if override:
        return int(override)
    normalized_dtype = _normalize_kernel_dtype(kernel_dtype)
    if normalized_dtype == "bfloat16":
        return 16
    return 8


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
        raise ValueError(
            f"Unsupported kernel dtype '{value}'. Expected one of: float32, float16, bfloat16."
        )
    return aliases[normalized]


def load_fused_kernel(
    head_dim: int = 64,
    tile_size: Optional[int] = None,
    kernel_dtype: Optional[str] = None,
    proj_k_tile: Optional[int] = None,
):
    """
    JIT-compile the CUDA extension and return the loaded module.

    Notes
    -----
    The kernel source is macro-parameterized by `HEAD_DIM` and `TILE_SIZE`, so
    PatchTST can request a head dimension that differs from the benchmark
    configuration.
    """
    global _kernel_cache
    if tile_size is None:
        tile_size = _default_tile_size()
    normalized_dtype = _normalize_kernel_dtype(kernel_dtype)
    if proj_k_tile is None:
        proj_k_tile = _default_proj_k_tile(normalized_dtype)
    cache_key = (int(head_dim), int(tile_size), int(proj_k_tile), normalized_dtype)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file = os.path.join(root, "kernel", "fused_attn.cu")
    cpp_file = os.path.join(root, "kernel", "fused_attn_ext.cpp")
    build_dir = os.path.join(root, "build")
    dtype_tag = "f32"
    if normalized_dtype == "float16":
        dtype_tag = "f16"
    elif normalized_dtype == "bfloat16":
        dtype_tag = "bf16"
    module_name = (
        f"fused_linear_attention_hd{head_dim}_tile{tile_size}_proj{proj_k_tile}_{dtype_tag}"
    )

    for path in (cu_file, cpp_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required kernel source not found: {path}\n"
                "The canonical kernel sources must live under the root kernel/ directory."
            )

    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _default_cuda_arch())

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        f"-DTILE_SIZE={int(tile_size)}",
        f"-DHEAD_DIM={int(head_dim)}",
        f"-DPROJ_K_TILE={int(proj_k_tile)}",
    ]

    if os.environ.get("FLA_ALLOW_UNSUPPORTED_COMPILER", "1") == "1":
        extra_cuda_cflags.append("-allow-unsupported-compiler")

    _kernel_cache[cache_key] = load(
        name=module_name,
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        build_directory=build_dir,
    )
    return _kernel_cache[cache_key]
