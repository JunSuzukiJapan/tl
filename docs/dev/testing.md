# Testing Guide

This document describes how to run tests for the TensorLogic (TL) compiler.

## Running Unit Tests

To run the standard Rust unit tests:

```bash
cargo test
```

## Running Verification Script

For comprehensive testing of all `.tl` files in the project (examples and tests), use the `verify_tl_files.py` script. This script executes each file and checks for runtime errors (non-zero exit codes).

### Basic Usage

Run all applicable `.tl` files sequentially:

```bash
python3 scripts/verify_tl_files.py
```

### Parallel Execution (Recommended)

To speed up execution, you can run tests in parallel using the `--parallel` option.

```bash
# Run with 8 parallel threads
python3 scripts/verify_tl_files.py --parallel 8
```

### Options

- `--parallel <N>`: Run N tests in parallel.
- `--verbose`: Show output for all tests, not just failures.

### Expected Failures

Some files in `tests/fixtures/patterns/*` are designed to fail (testing error reporting). The script may report these as failures or handle them if configured logic allows. Currently, the script reports non-zero exit codes as failures.

## Debugging Segfaults

If you encounter segmentation faults, please refer to [Debugging Segfaults](../.agent/workflows/debug-segfault.md) (or `.agent/workflows/debug-segfault.md` in the repository).
