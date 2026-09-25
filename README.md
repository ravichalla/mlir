# FP Precision-Lowering MLIR Pass + Differential Correctness Harness

A portfolio project demonstrating compiler transformation + numerical
verification, aimed at roles that combine compiler engineering with
correctness/verification infrastructure (e.g. NVIDIA's Agentic Compiler
Systems team).

## What this is

1. **`lib/PrecisionLoweringPass.cpp`** — an MLIR pass that walks `arith`
   dialect ops on `f32` operands and rewrites them to `f16`, inserting
   `arith.truncf` / `arith.extf` casts at the boundaries. This is a real
   (if intentionally scoped-down) optimization pass: this is exactly the
   kind of precision-lowering transformation that shows up in real
   ML-compiler pipelines to trade numerical range for throughput.

2. **`diff_test/`** — a differential testing harness that runs the *same*
   computation before and after the pass, on both CPU (via `mlir-cpu-runner`)
   and GPU (via a hand-written CUDA kernel pair), and flags correctness
   regressions using tolerance bounds tuned for fp16 rounding behavior
   (not just a flat epsilon — it separately tracks relative error,
   NaN/Inf propagation, and catastrophic cancellation in near-zero
   accumulations).

3. **`cuda/`** — a small matmul kernel implemented twice (fp32 baseline,
   fp16-with-fp32-accumulate) so the harness has a GPU-side correctness
   story too, not just an IR-level one.

## Why this shape (portfolio narrative)

Most "I built a compiler" portfolio projects stop at "it produces
different code." This one is scoped around the harder and more relevant
half of the problem: *proving the transformation didn't break anything*,
which is the actual emphasis of verification/differential-testing roles.
The precision-lowering pass is a good vehicle because it's simultaneously:

- a genuine optimization (perf/memory-bandwidth win on GPU workloads)
- inherently a correctness risk (this is where numerical bugs live)

so the same project produces both compiler and numerics talking points.

## Build requirements

- LLVM/MLIR built from source (or via a package that ships MLIR dev
  headers — Homebrew's `llvm` cask on macOS works; on Linux you generally
  need to build LLVM+MLIR yourself with `-DLLVM_ENABLE_PROJECTS=mlir`).
  This is the single biggest setup cost — budget a few hours for the
  LLVM build itself, separate from writing the pass.
- CMake >= 3.20, a C++17 compiler.
- CUDA toolkit + a GPU for the `cuda/` kernels. If you don't have local
  GPU access, cheap short-lived cloud GPU instances (Lambda Cloud, RunPod,
  Colab) are enough to run and record results for a portfolio writeup —
  you don't need sustained access.
- Python 3.9+, `numpy` for `diff_test/compare.py`.

## Build

```bash
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
                   -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm
ninja
```

## Run the pass on the sample IR

```bash
./build/precision-lower-opt test/sample.mlir -o /tmp/lowered.mlir
```

## Run the differential harness

```bash
python3 diff_test/harness.py \
    --original test/sample.mlir \
    --lowered /tmp/lowered.mlir \
    --rtol 1e-2 --atol 1e-3
```

## Status / what's stubbed

This is scaffolding meant to be filled in and actually built against a
real LLVM/MLIR checkout — it is **not** validated to compile as-is in
this environment (no MLIR toolchain or GPU available here). Treat the
`.cpp`/`.cu` files as a correct-by-construction starting point following
standard MLIR pass structure (`PassWrapper`, `OpRewritePattern`,
`ConversionTarget`), not as tested output. Before you put this on a
resume/portfolio, you'll need to:

1. Actually build it against an LLVM/MLIR checkout and fix whatever API
   drift exists between the MLIR version you build against and what's
   written here (MLIR's C++ API moves fast between releases).
2. Run the CUDA kernels on real hardware and capture real numbers.
3. Extend the pass to handle at least one more op class (e.g. `arith.mulf`
   accumulation chains) so it's not a single-op toy.
