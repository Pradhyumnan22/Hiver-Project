"""Tests for judge schemas, agreement metrics, and evaluation safeguards."""
import pytest
from pydantic import ValidationError

from src.evaluation.agreement import calculate_judge_human_agreement
from src.evaluation.judge import CriterionRating, JudgeRating


def rating_row(item_id, scores):
    criteria = (
        "groundedness",
        "relevance",
        "helpfulness",
        "consistency",
        "safety",
    )
    return {"id": item_id, **dict(zip(criteria, scores))}


def test_judge_rating_requires_scores_and_explanations():
    rating = JudgeRating(
        **{
            criterion: CriterionRating(score=5, explanation="Fully supported.")
            for criterion in (
                "groundedness",
                "relevance",
                "helpfulness",
                "consistency",
                "safety",
            )
        }
    )
    assert rating.mean_score == 5.0

    with pytest.raises(ValidationError):
        CriterionRating(score=6, explanation="Outside the scale.")
    with pytest.raises(ValidationError):
        CriterionRating(score=3, explanation="")


def test_judge_human_agreement_known_values():
    human = [
        rating_row("a", [1, 2, 3, 4, 5]),
        rating_row("b", [2, 3, 4, 5, 1]),
    ]
    judge = [
        rating_row("a", [1, 3, 3, 5, 4]),
        rating_row("b", [3, 3, 5, 5, 2]),
    ]

    result = calculate_judge_human_agreement(human, judge)

    assert result["shared_examples"] == 2
    assert result["per_criterion"]["groundedness"]["exact_agreement"] == 0.5
    assert result["per_criterion"]["groundedness"]["within_one_agreement"] == 1.0
    assert result["pooled"]["n"] == 10


def test_agreement_rejects_missing_overlap():
    with pytest.raises(ValueError, match="No shared IDs"):
        calculate_judge_human_agreement(
            [rating_row("human", [3, 3, 3, 3, 3])],
            [rating_row("judge", [3, 3, 3, 3, 3])],
        )


def test_agreement_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="1 to 5"):
        calculate_judge_human_agreement(
            [rating_row("a", [0, 3, 3, 3, 3])],
            [rating_row("a", [3, 3, 3, 3, 3])],
        )
