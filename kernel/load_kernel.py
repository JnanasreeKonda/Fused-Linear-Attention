"""
kernel/load_kernel.py — JIT-compile and load the canonical FusedLinearAttention extension.

This is the single source of truth for loading the CUDA kernel from the root
`kernel/` package. Scripts inside `baseline_pipeline/` should import this file
instead of reaching into any archived bundle directory.
"""

from __future__ import annotations

import os

_kernel_cache = {}
DEFAULT_TILE_SIZE = int(os.environ.get("FLA_TILE_SIZE", "32"))
DEFAULT_CUDA_ARCH = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0")


def load_fused_kernel(head_dim: int = 64, tile_size: int = DEFAULT_TILE_SIZE):
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
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_CUDA_ARCH)

    extra_cuda_cflags = [
        "-O3",
        "-arch=sm_80",
        "--use_fast_math",
        f"-DTILE_SIZE={int(tile_size)}",
        f"-DHEAD_DIM={int(head_dim)}",
    ]

    # Older cluster toolkits can reject newer host GCC versions by default.
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
