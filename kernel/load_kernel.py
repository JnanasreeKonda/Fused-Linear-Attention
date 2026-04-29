"""
kernel/load_kernel.py — JIT-compile and load the canonical FusedLinearAttention extension.

This is the single source of truth for loading the CUDA kernel from the root
`kernel/` package. Scripts inside `baseline_pipeline/` should import this file
instead of reaching into any archived bundle directory.
"""

from __future__ import annotations

import os

import torch

_kernel_cache = {}


def load_fused_kernel(head_dim: int = 64, tile_size: int = 64):
    """
    JIT-compile the CUDA extension and return the loaded module.

    Notes
    -----
    The kernel source is macro-parameterized by `HEAD_DIM` and `TILE_SIZE`, so
    PatchTST can request a head dimension that differs from the benchmark
    configuration.
    """
    global _kernel_cache
    cache_key = (int(head_dim), int(tile_size))
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file = os.path.join(root, "kernel", "fused_attn.cu")
    cpp_file = os.path.join(root, "kernel", "fused_attn_ext.cpp")
    build_dir = os.path.join(root, "build")
    module_name = f"fused_linear_attention_hd{head_dim}_tile{tile_size}"

    for path in (cu_file, cpp_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required kernel source not found: {path}\n"
                "The canonical kernel sources must live under the root kernel/ directory."
            )

    os.makedirs(build_dir, exist_ok=True)

    _kernel_cache[cache_key] = load(
        name=module_name,
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_80",
            "--use_fast_math",
            f"-DTILE_SIZE={int(tile_size)}",
            f"-DHEAD_DIM={int(head_dim)}",
        ],
        verbose=False,
        build_directory=build_dir,
    )
    return _kernel_cache[cache_key]
