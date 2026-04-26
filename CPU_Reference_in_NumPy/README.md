# `CPU_Reference_in_NumPy/`

This folder stores the original NumPy reference workflow and golden outputs
used for correctness validation.

## Contents

- `tests/reference.py`
  - original NumPy fused-attention implementation
- `tests/test_reference_vs_pytorch.py`
  - compares the NumPy implementation against PyTorch
- `tests/generate_golden_outputs.py`
  - generates saved golden outputs
- `tests/golden/`
  - trusted `.npy` inputs, weights, and outputs for selected test cases

## Role In The Repo

The canonical root-level `tests/` folder is the main correctness entrypoint for
the cleaned repo, while this folder preserves the original reference artifacts
and golden data used to build and validate that flow.
