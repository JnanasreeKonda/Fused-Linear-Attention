# `kernel/`

This folder contains the canonical fused-kernel implementation used by the
cleaned repository layout.

## Files

- `DESIGN.md`
  - tiling strategy, shared-memory budget, bank-conflict avoidance, and HBM
    traffic analysis
- `fused_attn.cu`
  - fused CUDA kernel implementation
- `fused_attn_ext.cpp`
  - PyTorch C++ extension binding
- `load_kernel.py`
  - JIT compilation and loading helper for the extension

## What Is Implemented

- a root-level fused-kernel code path
- PyTorch extension loading for the canonical kernel location
- benchmark-oriented `HEAD_DIM=64` configuration

## Current Limitation

The current fused path is still specialized around the benchmark kernel
configuration and is not yet fully generalized for the PatchTST model-side
`d_head=32` configuration.
