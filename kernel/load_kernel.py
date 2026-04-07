"""
kernel/load_kernel.py — JIT-compile and load the FusedLinearAttention CUDA extension.
Owner: Jnanasree  |  Milestone: M8  |  Phase 2

PLACEHOLDER — Jnanasree will implement this during Week 3.
"""

# TODO (Jnanasree, M8): implement load_fused_kernel() using torch.utils.cpp_extension.load()

def load_fused_kernel():
    raise NotImplementedError(
        "load_kernel.py is a Phase 2 deliverable (M8 — Jnanasree).\n"
        "It will JIT-compile kernel/fused_attn_ext.cpp + kernel/fused_attn.cu "
        "via torch.utils.cpp_extension.load()."
    )
