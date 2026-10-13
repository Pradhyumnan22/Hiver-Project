"""Tests verifying the golden evaluation set schema, size, intent coverage, and strict zero-leakage."""
import csv
import json
from pathlib import Path
import pytest
from src.data.intent_taxonomy import Action, EscalationReason, GoldenEvaluationExample, Intent

GOLDEN_SET_PATH = Path("data/processed/golden_eval_set.json")
DEV_PAIRS_PATH = Path("data/processed/dev_pairs.json")
REVIEW_QUEUE_PATH = Path("data/annotations/golden_review.csv")


@pytest.fixture
def golden_set():
    assert GOLDEN_SET_PATH.exists(), f"Golden set file not found at {GOLDEN_SET_PATH}"
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [GoldenEvaluationExample(**item) for item in data]


def test_golden_set_size_requirement(golden_set):
    """The assignment specifies a golden evaluation set of 150–250 examples."""
    assert 150 <= len(golden_set) <= 250, f"Expected 150-250 golden examples, got {len(golden_set)}"


def test_review_queue_has_200_unique_real_examples():
    with open(REVIEW_QUEUE_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 200
    assert len({row["id"] for row in rows}) == 200
    assert len({row["customer_tweet_id"] for row in rows}) == 200
    assert all(row["is_synthetic"] == "false" for row in rows)
    assert {row["expected_intent"] for row in rows} == {
        intent.value for intent in Intent
    }


def test_finalized_set_records_review_provenance(golden_set):
    assert all(item.annotation_method == "ai_manually_reviewed" for item in golden_set)
    assert all(item.reviewer_id for item in golden_set)
    assert all(item.reviewed_at for item in golden_set)
    assert len({item.customer_tweet_id for item in golden_set}) == len(golden_set)


def test_all_seven_intents_represented(golden_set):
    """Verify that all 7 defined taxonomy intents are present in the golden set."""
    intents_found = {item.expected_intent for item in golden_set}
    for intent in Intent:
        assert intent in intents_found, f"Missing intent: {intent.value} in golden evaluation set"


def test_escalation_rules_consistency(golden_set):
    """Verify that escalation reason matches expected action for 100% of golden examples."""
    for item in golden_set:
        if item.expected_action == Action.AUTO_HANDLE:
            assert item.escalation_reason == EscalationReason.NONE
        elif item.expected_action == Action.ESCALATE:
            assert item.escalation_reason != EscalationReason.NONE


def test_strict_zero_leakage_with_dev_set(golden_set):
    """Verify zero overlap between golden set customer tweets/authors and dev knowledge base."""
    if not DEV_PAIRS_PATH.exists():
        pytest.skip("dev_pairs.json not found for local run")

    with open(DEV_PAIRS_PATH, "r", encoding="utf-8") as f:
        dev_pairs = json.load(f)

    dev_author_ids = {p["customer_author_id"] for p in dev_pairs}
    dev_tweet_ids = {p["customer_tweet_id"] for p in dev_pairs}

    golden_authors = {item.customer_author_id for item in golden_set}
    golden_tweets = {item.customer_tweet_id for item in golden_set}

    author_overlap = dev_author_ids.intersection(golden_authors)
    tweet_overlap = dev_tweet_ids.intersection(golden_tweets)

    assert len(author_overlap) == 0, f"Critical leakage! Dev and Golden sets share authors: {author_overlap}"
    assert len(tweet_overlap) == 0, f"Critical leakage! Dev and Golden sets share tweets: {tweet_overlap}"
