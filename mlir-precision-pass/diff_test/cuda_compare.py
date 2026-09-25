#!/usr/bin/env python3
"""cuda_compare.py

Loads the fp32-baseline and fp16-compute outputs written by
cuda/diff_main.cu and runs them through the same compare.py verdict logic
used for the MLIR-side harness, so the portfolio writeup has one
correctness methodology covering both the IR-level and GPU-level halves
of the project.

Usage:
    ./matmul_diff 1024 /tmp/matmul
    python3 cuda_compare.py --prefix /tmp/matmul --n 1024 --rtol 5e-3 --atol 1e-2
"""

import argparse
import struct
import sys
from pathlib import Path

from compare import compare_arrays, summarize


def load_f32_binary(path: Path, count: int) -> list[float]:
    data = path.read_bytes()
    expected_bytes = count * 4
    if len(data) != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes ({count} f32 values), "
            f"got {len(data)}"
        )
    return list(struct.unpack(f"<{count}f", data))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True,
                     help="prefix passed to matmul_diff, e.g. /tmp/matmul")
    ap.add_argument("--n", type=int, required=True,
                     help="matrix dimension N used when running matmul_diff")
    # Looser than the MLIR-side default tolerances: a full NxN matmul
    # reduction accumulates rounding error over N terms, so per-element
    # tolerance needs to scale with N. Tighten these once you have real
    # hardware numbers to calibrate against rather than guessing.
    ap.add_argument("--rtol", type=float, default=5e-3)
    ap.add_argument("--atol", type=float, default=1e-2)
    args = ap.parse_args()

    n_values = args.n * args.n
    baseline = load_f32_binary(Path(f"{args.prefix}_fp32.bin"), n_values)
    lowered = load_f32_binary(Path(f"{args.prefix}_fp16.bin"), n_values)

    results = compare_arrays(baseline, lowered, rtol=args.rtol, atol=args.atol)
    print(summarize(results))

    return 0 if all(r.within_tolerance for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
