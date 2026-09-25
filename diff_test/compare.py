"""compare.py

Shared numerical-verdict logic used by both the MLIR-side harness
(harness.py) and the CUDA-side comparison (cuda_compare.py). Kept in one
place so "what counts as a correctness regression" is defined once.

The core idea: a flat epsilon is the wrong tool for comparing fp32 vs
fp16-lowered results, because fp16 error is inherently relative (it scales
with magnitude) and blows up specifically near cancellation. So this
module reports three separate signals instead of one pass/fail bit:

  1. Relative + absolute tolerance check (standard, like np.allclose)
  2. NaN/Inf propagation mismatches (did lowering introduce or hide these?)
  3. Catastrophic cancellation flags (large operands, small result --
     exactly where fp16 rounding error, which is bounded in *relative*
     terms on the operands, becomes large in *relative* terms on the
     result)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence


@dataclasses.dataclass
class ComparisonResult:
    index: int
    baseline: float
    lowered: float
    within_tolerance: bool
    relative_error: float | None
    nan_inf_mismatch: bool
    cancellation_flag: bool
    note: str = ""


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def compare_scalar(
    index: int,
    baseline: float,
    lowered: float,
    *,
    rtol: float,
    atol: float,
    operand_magnitude_hint: float | None = None,
    cancellation_ratio_threshold: float = 1e3,
) -> ComparisonResult:
    """Compare one (baseline, lowered) pair and classify the result.

    operand_magnitude_hint: if the caller knows the magnitude of the
    inputs that fed this result (e.g. from the near_cancellation test
    case), pass it so we can flag cancellation even when the *result*
    itself is small -- that's precisely the case a flat tolerance check
    would silently pass or silently fail without explanation.
    """
    baseline_finite = _is_finite(baseline)
    lowered_finite = _is_finite(lowered)
    nan_inf_mismatch = baseline_finite != lowered_finite

    if nan_inf_mismatch:
        return ComparisonResult(
            index=index,
            baseline=baseline,
            lowered=lowered,
            within_tolerance=False,
            relative_error=None,
            nan_inf_mismatch=True,
            cancellation_flag=False,
            note="NaN/Inf status differs between baseline and lowered result",
        )

    if not baseline_finite:
        # Both NaN/Inf and agree -- treat as a match, nothing more to check.
        return ComparisonResult(
            index=index,
            baseline=baseline,
            lowered=lowered,
            within_tolerance=True,
            relative_error=None,
            nan_inf_mismatch=False,
            cancellation_flag=False,
            note="both non-finite and agree",
        )

    abs_err = abs(baseline - lowered)
    denom = max(abs(baseline), 1e-12)
    rel_err = abs_err / denom

    within = abs_err <= atol + rtol * abs(baseline)

    cancellation_flag = False
    if operand_magnitude_hint is not None and operand_magnitude_hint > 0:
        ratio = operand_magnitude_hint / max(abs(baseline), 1e-12)
        if ratio > cancellation_ratio_threshold:
            cancellation_flag = True

    note = ""
    if cancellation_flag and within:
        note = (
            "passed tolerance but operand/result magnitude ratio "
            f"({operand_magnitude_hint:.3g}/{abs(baseline):.3g}) suggests "
            "near-cancellation -- verify with a wider tolerance sweep before "
            "trusting this result"
        )
    elif cancellation_flag and not within:
        note = "failure at a near-cancellation site -- expected failure mode, not a surprise"

    return ComparisonResult(
        index=index,
        baseline=baseline,
        lowered=lowered,
        within_tolerance=within,
        relative_error=rel_err,
        nan_inf_mismatch=False,
        cancellation_flag=cancellation_flag,
        note=note,
    )


def compare_arrays(
    baseline: Sequence[float],
    lowered: Sequence[float],
    *,
    rtol: float,
    atol: float,
) -> list[ComparisonResult]:
    if len(baseline) != len(lowered):
        raise ValueError(
            f"length mismatch: baseline has {len(baseline)} values, "
            f"lowered has {len(lowered)}"
        )
    return [
        compare_scalar(i, b, l, rtol=rtol, atol=atol)
        for i, (b, l) in enumerate(zip(baseline, lowered))
    ]


def summarize(results: Sequence[ComparisonResult]) -> str:
    total = len(results)
    failures = [r for r in results if not r.within_tolerance]
    nan_mismatches = [r for r in results if r.nan_inf_mismatch]
    cancellations = [r for r in results if r.cancellation_flag]

    lines = [
        f"{total} values compared, {len(failures)} tolerance failures, "
        f"{len(nan_mismatches)} NaN/Inf mismatches, "
        f"{len(cancellations)} near-cancellation sites flagged.",
    ]
    for r in failures[:20]:
        lines.append(
            f"  [{r.index}] baseline={r.baseline!r} lowered={r.lowered!r} "
            f"rel_err={r.relative_error} note={r.note}"
        )
    if len(failures) > 20:
        lines.append(f"  ... and {len(failures) - 20} more failures")
    return "\n".join(lines)
