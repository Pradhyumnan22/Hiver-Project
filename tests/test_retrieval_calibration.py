"""Tests for deterministic retrieval-threshold calibration."""
import pytest

from src.knowledge.calibration import calibrate_threshold


def test_calibration_selects_highest_coverage_eligible_threshold():
    result = calibrate_threshold(
        scores=[0.90, 0.80, 0.70, 0.60, 0.50],
        relevant=[True, True, False, True, False],
        candidates=[0.50, 0.60, 0.70, 0.80],
        target_precision=0.75,
        minimum_coverage=0.20,
    )

    assert result.selected_threshold == 0.6
    selected = next(
        metric
        for metric in result.metrics
        if metric.threshold == result.selected_threshold
    )
    assert selected.coverage == 0.8
    assert selected.proxy_precision == 0.75


def test_calibration_rejects_impossible_constraints():
    with pytest.raises(ValueError, match="No threshold"):
        calibrate_threshold(
            scores=[0.9, 0.8],
            relevant=[False, False],
            candidates=[0.5, 0.8],
            target_precision=0.8,
            minimum_coverage=0.5,
        )
