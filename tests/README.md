# `tests/`

This folder contains the canonical root-level correctness checks for the
cleaned repository layout.

## Files

- `reference.py`
  - root-level NumPy oracle used by the consolidated repo
- `test_correctness.py`
  - fused-kernel correctness suite
- `test_fused_backward.py`
  - smoke-test for the custom autograd backward bridge

## Purpose

These tests provide the single source of truth for correctness checks in the
main repo layout. They are intended to compare the fused implementation against
trusted NumPy and PyTorch reference behavior.

## Typical Commands

Quick simulation:

```bash
python3 tests/test_correctness.py --simulate --quick --use-root-oracle
python3 tests/test_fused_backward.py
```

Quick CUDA-backed check:

```bash
python3 tests/test_correctness.py --quick
```

Full suite:

```bash
python3 tests/test_correctness.py
```

## Notes

- `test_correctness.py --simulate` validates the comparison logic without a
  compiled CUDA kernel.
- `test_fused_backward.py` validates the custom backward bridge without
  requiring CUDA by using a fake kernel forward that matches the reference
  attention math.
- GPU-backed fused-kernel validation still depends on a working CUDA toolchain
  and supported runtime environment.
