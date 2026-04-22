"""
kernel/load_kernel.py — JIT-compile and load the FusedLinearAttention CUDA extension.
Owner: Jnanasree (M8) — this file is provided here for Bhanuja's fused_bench.py to work.

Compiles:
  kernel/fused_attn.cu       (Bhanuja v3)
  kernel/fused_attn_ext.cpp  (Jnanasree M8 — PyTorch C++ binding)

Usage:
    from kernel.load_kernel import load_fused_kernel
    fused = load_fused_kernel()
    out   = fused.forward(X, Wq, Wk, Wv, B, H, N, D, d_head)
    # out: torch.Tensor [B, H, N, d_head]
"""

import os
import torch

_kernel_cache = None


def load_fused_kernel():
    """
    JIT-compile the CUDA extension and return the loaded module.
    Result is cached — only compiles once per Python process.
    """
    global _kernel_cache
    if _kernel_cache is not None:
        return _kernel_cache

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_file  = os.path.join(root, "kernel", "fused_attn.cu")
    cpp_file = os.path.join(root, "kernel", "fused_attn_ext.cpp")

    if not os.path.exists(cpp_file):
        raise FileNotFoundError(
            f"fused_attn_ext.cpp not found at {cpp_file}.\n"
            "This is Jnanasree's M8 deliverable. "
            "Run fused_bench.py --simulate for wall-time without the compiled kernel."
        )

    _kernel_cache = load(
        name="fused_linear_attention",
        sources=[cpp_file, cu_file],
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_80",          # A100
            "--use_fast_math",
            f"-DTILE_SIZE=64",
            f"-DHEAD_DIM=64",
        ],
        verbose=False,
        build_directory=os.path.join(root, "build"),
    )
    return _kernel_cache
