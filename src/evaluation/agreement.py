"""Judge–human rating agreement calculations."""
import math
from typing import Sequence

from sklearn.metrics import cohen_kappa_score

from src.evaluation.judge import JUDGE_CRITERIA


def _agreement_summary(human: Sequence[int], judge: Sequence[int]) -> dict:
    if len(human) != len(judge) or not human:
        raise ValueError("human and judge ratings must be non-empty and aligned.")
    if any(score < 1 or score > 5 for score in [*human, *judge]):
        raise ValueError("All rubric scores must be integers from 1 to 5.")

    exact = sum(h == j for h, j in zip(human, judge)) / len(human)
    within_one = sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / len(human)
    kappa = float(cohen_kappa_score(human, judge, weights="quadratic"))
    return {
        "n": len(human),
        "exact_agreement": round(exact, 4),
        "within_one_agreement": round(within_one, 4),
        "weighted_cohens_kappa": None if math.isnan(kappa) else round(kappa, 4),
    }


def calculate_judge_human_agreement(
    human_rows: Sequence[dict],
    judge_rows: Sequence[dict],
) -> dict:
    """Align ratings by ID and calculate criterion and pooled agreement."""
    human_by_id = {str(row["id"]): row for row in human_rows}
    judge_by_id = {str(row["id"]): row for row in judge_rows}
    shared_ids = sorted(set(human_by_id) & set(judge_by_id))
    if not shared_ids:
        raise ValueError("No shared IDs between human and judge ratings.")

    per_criterion = {}
    pooled_human = []
    pooled_judge = []
    for criterion in JUDGE_CRITERIA:
        human_scores = [int(human_by_id[item][criterion]) for item in shared_ids]
        judge_scores = [int(judge_by_id[item][criterion]) for item in shared_ids]
        per_criterion[criterion] = _agreement_summary(
            human_scores,
            judge_scores,
        )
        pooled_human.extend(human_scores)
        pooled_judge.extend(judge_scores)

    return {
        "shared_examples": len(shared_ids),
        "criteria": list(JUDGE_CRITERIA),
        "per_criterion": per_criterion,
        "pooled": _agreement_summary(pooled_human, pooled_judge),
    }
