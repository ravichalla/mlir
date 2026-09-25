"""test_compare.py

Unit tests for the comparison logic itself -- worth having in this
project specifically because compare.py is the thing that decides
pass/fail for everything else; a bug here would silently validate a
broken pass. Run with: python3 -m pytest diff_test/test_compare.py
"""

import math

from compare import compare_scalar


def test_exact_match():
    r = compare_scalar(0, 1.0, 1.0, rtol=1e-3, atol=1e-6)
    assert r.within_tolerance
    assert r.relative_error == 0.0


def test_within_tolerance():
    r = compare_scalar(0, 100.0, 100.5, rtol=1e-2, atol=1e-3)
    assert r.within_tolerance


def test_outside_tolerance():
    r = compare_scalar(0, 100.0, 110.0, rtol=1e-2, atol=1e-3)
    assert not r.within_tolerance


def test_nan_mismatch_flagged():
    r = compare_scalar(0, 1.0, float("nan"), rtol=1e-2, atol=1e-3)
    assert r.nan_inf_mismatch
    assert not r.within_tolerance


def test_both_nan_is_a_match():
    r = compare_scalar(0, float("nan"), float("nan"), rtol=1e-2, atol=1e-3)
    assert not r.nan_inf_mismatch
    assert r.within_tolerance


def test_inf_mismatch_flagged():
    r = compare_scalar(0, float("inf"), 1e30, rtol=1e-2, atol=1e-3)
    assert r.nan_inf_mismatch


def test_cancellation_flag_on_large_operand_small_result():
    # Mirrors test/sample.mlir's near_cancellation case: operands ~1000,
    # result ~0.06 -- the ratio should trip the cancellation flag even
    # though this specific pair happens to be within tolerance.
    r = compare_scalar(
        0, 0.0625, 0.06, rtol=1e-1, atol=1e-2, operand_magnitude_hint=1000.0
    )
    assert r.cancellation_flag


def test_no_cancellation_flag_without_hint():
    r = compare_scalar(0, 0.0625, 0.06, rtol=1e-1, atol=1e-2)
    assert not r.cancellation_flag


if __name__ == "__main__":
    # Allow running without pytest installed.
    import sys

    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            print(f"FAIL {t.__name__}")
            failed += 1
    sys.exit(1 if failed else 0)
