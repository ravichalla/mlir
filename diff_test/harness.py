#!/usr/bin/env python3
"""harness.py

Differential correctness harness for the precision-lowering pass.

Runs the *same* MLIR function twice -- once as originally written (f32
throughout), once after the precision-lowering pass -- via mlir-cpu-runner,
and compares outputs using compare.py's tolerance/NaN/cancellation logic.

This is the piece of the project that answers "how do you know the
transformation is correct," which is the actual point of a verification-
flavored compiler role, as opposed to just "the pass runs and produces
different IR."

Usage:
    python3 harness.py --original test/sample.mlir \
                        --lowered /tmp/lowered.mlir \
                        --rtol 1e-2 --atol 1e-3

Requires: mlir-cpu-runner and mlir-translate on PATH (both ship with an
MLIR build). NOT validated to run in this environment -- no MLIR
toolchain available here. Treat the subprocess plumbing as a correct
starting point to adapt to your actual mlir-cpu-runner invocation
(entry-point name / result printing flags vary slightly by MLIR version).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from compare import compare_arrays, summarize


def run_via_cpu_runner(mlir_file: Path, entry_point: str) -> list[float]:
    """Lower to LLVM dialect, JIT it via mlir-cpu-runner, and parse the
    printed float results.

    In a real build this pipes through `mlir-opt --convert-arith-to-llvm
    --convert-func-to-llvm ... | mlir-translate --mlir-to-llvmir | ...`
    or, more simply for f32-scalar-returning functions like these test
    cases, wraps the function with a small `func.func @main` that calls
    it and prints the result via `vector.print` / `func.call @printF32`,
    then runs `mlir-cpu-runner --entry-point-result=f32`.

    Wiring the exact lowering pipeline is toolchain/version-specific
    enough that it's left as the first thing to fill in once you have a
    real MLIR build -- the rest of the harness (comparison, reporting)
    doesn't depend on these details.
    """
    cmd = [
        "mlir-cpu-runner",
        str(mlir_file),
        "--entry-point-result=f32",
        f"--entry-point={entry_point}",
        "-shared-libs=libmlir_runner_utils.so,libmlir_c_runner_utils.so",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mlir-cpu-runner failed on {mlir_file} ({entry_point}):\n"
            f"{proc.stderr}"
        )
    # mlir-cpu-runner prints the raw result value; parse any floats found.
    values = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?",
                                            proc.stdout)]
    return values


def discover_entry_points(mlir_file: Path) -> list[str]:
    """Pull func.func names out of the IR so the harness runs every
    function in the file without the caller having to list them."""
    text = mlir_file.read_text()
    return re.findall(r"func\.func\s+@(\w+)", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--lowered", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    entry_points = discover_entry_points(args.original)
    if not entry_points:
        print(f"No func.func entries found in {args.original}", file=sys.stderr)
        return 1

    overall_ok = True
    for entry in entry_points:
        try:
            baseline_vals = run_via_cpu_runner(args.original, entry)
            lowered_vals = run_via_cpu_runner(args.lowered, entry)
        except RuntimeError as e:
            print(f"[{entry}] ERROR: {e}", file=sys.stderr)
            overall_ok = False
            continue

        results = compare_arrays(
            baseline_vals, lowered_vals, rtol=args.rtol, atol=args.atol
        )
        print(f"=== {entry} ===")
        print(summarize(results))
        if any(not r.within_tolerance for r in results):
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
