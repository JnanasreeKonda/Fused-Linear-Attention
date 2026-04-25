"""
kernel/load_kernel.py — JIT-compile and load the canonical FusedLinearAttention extension.

This is the single source of truth for loading the CUDA kernel from the root
`kernel/` package. Scripts inside `baseline_pipeline/` should import this file
instead of reaching into any archived bundle directory.
"""

from __future__ import annotations

import os

import torch

_kernel_cache = None


def load_fused_kernel():
    """
    JIT-compile the CUDA extension and return the loaded module.

    Notes
    -----
    The current kernel is compiled for the benchmark configuration
    `TILE_SIZE=64, HEAD_DIM=64` on `sm_80` (A100). The integration wrapper in
    `baseline_pipeline/model/fused_attn_block.py` guards against unsupported
    head dimensions.
    """
    global _kernel_cache
    if _kernel_cache is not None:
        return _kernel_cache

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file = os.path.join(root, "kernel", "fused_attn.cu")
    cpp_file = os.path.join(root, "kernel", "fused_attn_ext.cpp")
    build_dir = os.path.join(root, "build")

    for path in (cu_file, cpp_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required kernel source not found: {path}\n"
                "The canonical kernel sources must live under the root kernel/ directory."
            )

    os.makedirs(build_dir, exist_ok=True)

    _kernel_cache = load(
        name="fused_linear_attention",
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_80",
            "--use_fast_math",
            "-DTILE_SIZE=64",
            "-DHEAD_DIM=64",
        ],
        verbose=False,
        build_directory=build_dir,
    )
    return _kernel_cache
