"""
kernel/load_wmma_proj.py — JIT loader for the experimental WMMA projection extension.
"""

from __future__ import annotations

import os

_wmma_cache = None


def load_wmma_projection():
    global _wmma_cache
    if _wmma_cache is not None:
        return _wmma_cache

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file = os.path.join(root, "kernel", "wmma_proj.cu")
    cpp_file = os.path.join(root, "kernel", "wmma_proj_ext.cpp")
    build_dir = os.path.join(root, "build", "wmma_proj")
    os.makedirs(build_dir, exist_ok=True)

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ]

    _wmma_cache = load(
        name="wmma_projection_fp16",
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        build_directory=build_dir,
    )
    return _wmma_cache
