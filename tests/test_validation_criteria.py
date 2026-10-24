"""Native BF16 batch variance must not be misclassified as a parallel error."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "parallel_validation", Path(__file__).with_name("run_real_parallel_validation.py")
)
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


def test_matching_native_batch_semantics_passes():
    assert VALIDATION.passed(
        {
            "numerical_pass": True,
            "routing_exact": True,
            "top1_exact": True,
            "batch_vs_solo_pass": False,
            "batch_semantics_exact": True,
        }
    )


@pytest.mark.parametrize(
    "failed_check",
    ["numerical_pass", "routing_exact", "top1_exact", "batch_semantics_exact"],
)
def test_any_native_mismatch_fails(failed_check):
    checks = {
        "numerical_pass": True,
        "routing_exact": True,
        "top1_exact": True,
        "batch_vs_solo_pass": True,
        "batch_semantics_exact": True,
    }
    checks[failed_check] = False
    assert not VALIDATION.passed(checks)


def test_reference_mode_requires_bitwise_equality():
    report = {
        "numerical_pass": True,
        "routing_exact": True,
        "top1_exact": True,
        "batch_semantics_exact": True,
        "parallel_numerics": "reference",
        "bitwise_exact": False,
    }
    assert not VALIDATION.passed(report)
    report["bitwise_exact"] = True
    assert VALIDATION.passed(report)


def test_reference_mode_without_equality_evidence_fails():
    report = {
        "numerical_pass": True,
        "routing_exact": True,
        "top1_exact": True,
        "batch_semantics_exact": True,
        "parallel_numerics": "reference",
    }
    assert not VALIDATION.passed(report)
