# `tests/`

This folder contains the canonical root-level correctness checks for the
cleaned repository layout.

## Files

- `reference.py`
  - root-level NumPy oracle used by the consolidated repo
- `test_correctness.py`
  - fused-kernel correctness suite

## Purpose

These tests provide the single source of truth for correctness checks in the
main repo layout. They are intended to compare the fused implementation against
trusted NumPy and PyTorch reference behavior.

## Notes

- `test_correctness.py --simulate` validates the comparison logic without a
  compiled CUDA kernel.
- GPU-backed fused-kernel validation still depends on a working CUDA toolchain
  and supported runtime environment.
