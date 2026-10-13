"""Development-only calibration utilities for retrieval relevance thresholds."""
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ThresholdMetric:
    threshold: float
    accepted: int
    coverage: float
    proxy_precision: float


@dataclass(frozen=True)
class CalibrationResult:
    selected_threshold: float
    target_precision: float
    minimum_coverage: float
    metrics: tuple[ThresholdMetric, ...]


def calibrate_threshold(
    scores: Sequence[float],
    relevant: Sequence[bool],
    candidates: Iterable[float],
    target_precision: float = 0.80,
    minimum_coverage: float = 0.10,
) -> CalibrationResult:
    """Select the highest-coverage threshold meeting development constraints.

    ``relevant`` is a development-only proxy label. In this project it means
    that the nearest historical case has the same weakly supervised intent as
    the held-out calibration query; it is not a human relevance judgement.
    """
    if len(scores) != len(relevant) or not scores:
        raise ValueError("scores and relevant must be non-empty and equal length.")

    metrics = []
    total = len(scores)
    for threshold in sorted(set(candidates)):
        accepted_labels = [
            is_relevant
            for score, is_relevant in zip(scores, relevant)
            if score >= threshold
        ]
        accepted = len(accepted_labels)
        coverage = accepted / total
        precision = (
            sum(accepted_labels) / accepted if accepted else 0.0
        )
        metrics.append(
            ThresholdMetric(
                threshold=round(float(threshold), 4),
                accepted=accepted,
                coverage=round(coverage, 4),
                proxy_precision=round(precision, 4),
            )
        )

    eligible = [
        metric
        for metric in metrics
        if metric.proxy_precision >= target_precision
        and metric.coverage >= minimum_coverage
    ]
    if not eligible:
        raise ValueError(
            "No threshold satisfies the target precision and minimum coverage."
        )

    selected = max(
        eligible,
        key=lambda metric: (metric.coverage, metric.proxy_precision, metric.threshold),
    )
    return CalibrationResult(
        selected_threshold=selected.threshold,
        target_precision=target_precision,
        minimum_coverage=minimum_coverage,
        metrics=tuple(metrics),
    )
