"""Prepare and finalize a provenance-tracked golden evaluation set.

Preparation creates a deterministic, de-duplicated CSV review queue with
preliminary heuristic labels. Finalization refuses to publish the JSON golden
set until every row has explicit semantic-review provenance.
"""
import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from src.data.intent_taxonomy import Action, EscalationReason, GoldenEvaluationExample, Intent
from src.utils.logging import setup_logger

logger = setup_logger("build_golden_set")

EVAL_PAIRS_PATH = Path("data/processed/eval_pairs.json")
OUTPUT_GOLDEN_PATH = Path("data/processed/golden_eval_set.json")
REVIEW_QUEUE_PATH = Path("data/annotations/golden_review.csv")
REVIEW_BATCH_DIR = Path("data/annotations/reviews")
TARGET_SIZE = 200
SEED = 42


def classify_candidate(text: str, brand_reply: str):
    """Heuristic rule system based on the annotation guide to map candidate examples."""
    t = text.lower()
    b = brand_reply.lower()

    # 1. account_access_security
    if any(k in t for k in ["login", "log in", "password", "otp", "2fa", "unauthorized", "hacked", "close my account", "closed", "security code", "compromised", "charge on my card"]):
        return (
            Intent.ACCOUNT_ACCESS_SECURITY,
            Action.ESCALATE,
            EscalationReason.ACCOUNT_SECURITY_PII,
            "Customer account/security issue requires secure verification.",
        )

    # 2. subscription_digital
    if any(k in t for k in ["prime", "prime video", "kindle", "audible", "subscription", "membership", "free trial", "renew"]):
        if any(k in t for k in ["charged", "cancel", "refund", "unauthorized"]):
            return (
                Intent.SUBSCRIPTION_DIGITAL,
                Action.ESCALATE,
                EscalationReason.REFUND_REPLACEMENT_DISPUTE,
                "Subscription billing dispute or cancellation requiring account change.",
            )
        return (
            Intent.SUBSCRIPTION_DIGITAL,
            Action.AUTO_HANDLE,
            EscalationReason.NONE,
            "General inquiry about subscription benefits or features.",
        )

    # 3. damaged_defective_item
    if any(k in t for k in ["damaged", "broken", "cracked", "shattered", "defective", "stopped working", "wrong item", "missing piece", "ruined", "expired"]):
        return (
            Intent.DAMAGED_DEFECTIVE_ITEM,
            Action.ESCALATE,
            EscalationReason.REFUND_REPLACEMENT_DISPUTE,
            "Damaged/defective product reported; requires replacement/refund lookup.",
        )

    # 4. returns_refunds
    if any(k in t for k in ["return", "refund", "send back", "drop off", "ups drop", "return label", "reimburse", "return window"]):
        if any(k in t for k in ["where is my refund", "haven't received refund", "money back", "status of my return"]):
            return (
                Intent.RETURNS_REFUNDS,
                Action.ESCALATE,
                EscalationReason.REFUND_REPLACEMENT_DISPUTE,
                "Specific refund timeline or return tracking dispute requiring order lookup.",
            )
        return (
            Intent.RETURNS_REFUNDS,
            Action.AUTO_HANDLE,
            EscalationReason.NONE,
            "Standard return instructions or policy guidance.",
        )

    # 5. order_status_delivery
    if any(k in t for k in ["delivery", "delivered", "package", "tracking", "track", "courier", "driver", "late", "arrived", "carrier", "shipment", "shipped", "where is my order", "order id", "eta"]):
        if any(k in t for k in ["says delivered", "not here", "missing", "stolen", "late", "never showed", "delayed"]):
            return (
                Intent.ORDER_STATUS_DELIVERY,
                Action.ESCALATE,
                EscalationReason.ORDER_LOOKUP_REQUIRED,
                "Missing or severely delayed delivery requiring internal tracking lookup.",
            )
        return (
            Intent.ORDER_STATUS_DELIVERY,
            Action.AUTO_HANDLE,
            EscalationReason.NONE,
            "Standard delivery window or public tracking guidance.",
        )

    # 6. product_inquiry_availability
    if any(k in t for k in ["in stock", "available", "restock", "specifications", "compatible", "warranty", "price match", "how much is"]):
        return (
            Intent.PRODUCT_INQUIRY_AVAILABILITY,
            Action.AUTO_HANDLE,
            EscalationReason.NONE,
            "Product information or stock inquiry answerable from public catalog.",
        )

    # 7. other_unknown
    if len(t.split()) < 5 or any(k in t for k in ["thank", "thanks", "hello", "hi", "good morning", "awesome", "great"]):
        return (
            Intent.OTHER_UNKNOWN,
            Action.AUTO_HANDLE,
            EscalationReason.NONE,
            "General greeting, feedback, or conversational pleasantry.",
        )

    return (
        Intent.OTHER_UNKNOWN,
        Action.ESCALATE,
        EscalationReason.AMBIGUOUS_UNSUPPORTED,
        "Ambiguous or out-of-scope customer inquiry requiring human clarification.",
    )


def prepare_review_queue() -> None:
    """Create a deterministic review queue without claiming human verification."""
    logger.info(f"Loading held-out evaluation pairs from {EVAL_PAIRS_PATH}")
    with open(EVAL_PAIRS_PATH, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)

    # A customer tweet can have multiple brand replies. Keep one source row per
    # customer message so evaluation metrics cannot count the same input twice.
    unique_pairs = {}
    for pair in eval_pairs:
        unique_pairs.setdefault(int(pair["customer_tweet_id"]), pair)

    buckets = {intent: [] for intent in Intent}
    for p in unique_pairs.values():
        text = p["cleaned_customer_text"]
        brand_reply = p["cleaned_brand_text"]
        intent, action, reason, notes = classify_candidate(text, brand_reply)
        buckets[intent].append(
            {
                "id": f"gold_{p['customer_tweet_id']}",
                "source_pair_id": p["pair_id"],
                "customer_tweet_id": p["customer_tweet_id"],
                "brand_tweet_id": p["brand_tweet_id"],
                "customer_author_id": p["customer_author_id"],
                "cleaned_customer_message": p["cleaned_customer_text"],
                "conversation_context": (
                    "Direct customer-to-brand turn; no earlier conversation "
                    "turn was preserved by the source-pair extraction."
                ),
                "historical_brand_reply": p["cleaned_brand_text"],
                "expected_intent": intent.value,
                "expected_action": action.value,
                "escalation_reason": reason.value,
                "annotation_notes": notes,
                "is_synthetic": "false",
                "review_status": "pending",
                "reviewer_id": "",
                "reviewed_at": "",
                "review_notes": "",
            }
        )

    # Target: 200 total examples balanced across all 7 intents
    # Minimum 15 per intent, with larger allocations for primary customer contact drivers
    allocations = {
        Intent.ORDER_STATUS_DELIVERY: 45,
        Intent.RETURNS_REFUNDS: 35,
        Intent.DAMAGED_DEFECTIVE_ITEM: 30,
        Intent.ACCOUNT_ACCESS_SECURITY: 25,
        Intent.SUBSCRIPTION_DIGITAL: 25,
        Intent.PRODUCT_INQUIRY_AVAILABILITY: 20,
        Intent.OTHER_UNKNOWN: 20,
    }

    rng = random.Random(SEED)
    for available in buckets.values():
        available.sort(key=lambda row: int(row["customer_tweet_id"]))
        rng.shuffle(available)

    selected_examples: List[dict] = []
    selected_ids = set()
    for intent, count in allocations.items():
        available = buckets[intent]
        chosen = available[:count]
        selected_examples.extend(chosen)
        selected_ids.update(row["customer_tweet_id"] for row in chosen)
        if len(chosen) < count:
            logger.warning(
                f"Intent {intent.value} has {len(chosen)} unique candidates "
                f"for a target of {count}; shortfall will be backfilled."
            )

    # Backfill sparse intent allocations while retaining one row per tweet.
    remainder = [
        row
        for available in buckets.values()
        for row in available
        if row["customer_tweet_id"] not in selected_ids
    ]
    rng.shuffle(remainder)
    selected_examples.extend(remainder[: TARGET_SIZE - len(selected_examples)])
    selected_examples.sort(key=lambda row: (row["expected_intent"], int(row["customer_tweet_id"])))

    if len(selected_examples) != TARGET_SIZE:
        raise ValueError(
            f"Could only prepare {len(selected_examples)} unique examples; "
            f"{TARGET_SIZE} are required."
        )

    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_QUEUE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_examples[0]))
        writer.writeheader()
        writer.writerows(selected_examples)
    logger.info(
        f"Prepared {len(selected_examples)} unique examples at {REVIEW_QUEUE_PATH}. "
        "Labels are preliminary; semantic review is still required."
    )


def finalize_golden_set() -> None:
    """Publish reviewed rows only after validating review provenance."""
    with open(REVIEW_QUEUE_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != TARGET_SIZE:
        raise ValueError(f"Expected exactly {TARGET_SIZE} review rows, got {len(rows)}.")
    tweet_ids = [int(row["customer_tweet_id"]) for row in rows]
    if len(set(tweet_ids)) != TARGET_SIZE:
        raise ValueError("Each golden example must have a unique customer_tweet_id.")

    accepted_statuses = {"verified", "ai_reviewed"}
    pending = [
        row["id"]
        for row in rows
        if row["review_status"].strip().lower() not in accepted_statuses
        or not row["reviewer_id"].strip()
        or not row["reviewed_at"].strip()
    ]
    if pending:
        raise ValueError(
            f"{len(pending)} rows still require semantic review, reviewer_id, "
            f"and reviewed_at; first pending IDs: {pending[:5]}"
        )

    validated = []
    for row in rows:
        example = GoldenEvaluationExample(
            id=row["id"],
            source_pair_id=row["source_pair_id"],
            customer_tweet_id=int(row["customer_tweet_id"]),
            brand_tweet_id=int(row["brand_tweet_id"]),
            customer_author_id=row["customer_author_id"],
            customer_message=row["cleaned_customer_message"],
            cleaned_customer_message=row["cleaned_customer_message"],
            conversation_context=row["conversation_context"],
            expected_intent=row["expected_intent"],
            expected_action=row["expected_action"],
            escalation_reason=row["escalation_reason"],
            annotation_notes=row["annotation_notes"],
            historical_brand_reply=row["historical_brand_reply"],
            is_synthetic=row["is_synthetic"].strip().lower() == "true",
            annotation_method=(
                "human_reviewed"
                if row["review_status"].strip().lower() == "verified"
                else "ai_manually_reviewed"
            ),
            reviewer_id=row["reviewer_id"].strip(),
            reviewed_at=row["reviewed_at"].strip(),
            review_notes=row["review_notes"].strip() or None,
        )
        validated.append(example.model_dump(mode="json"))

    OUTPUT_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
    logger.info(f"Published {len(validated)} reviewed examples to {OUTPUT_GOLDEN_PATH}")


def apply_ai_reviews() -> None:
    """Merge independently reviewed batch files into the annotation audit CSV."""
    with open(REVIEW_QUEUE_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0])

    batch_paths = sorted(REVIEW_BATCH_DIR.glob("batch_*.json"))
    reviews = {}
    for path in batch_paths:
        with open(path, "r", encoding="utf-8") as f:
            for review in json.load(f):
                review_id = review["id"]
                if review_id in reviews:
                    raise ValueError(f"Duplicate reviewed ID: {review_id}")
                reviews[review_id] = review

    queue_ids = {row["id"] for row in rows}
    if set(reviews) != queue_ids:
        missing = sorted(queue_ids - set(reviews))
        extra = sorted(set(reviews) - queue_ids)
        raise ValueError(f"Review coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    editable_fields = (
        "expected_intent",
        "expected_action",
        "escalation_reason",
        "annotation_notes",
        "review_notes",
    )
    for row in rows:
        review = reviews[row["id"]]
        for field in editable_fields:
            row[field] = review[field]
        # Validate taxonomy and action/reason consistency before updating the CSV.
        GoldenEvaluationExample(
            id=row["id"],
            customer_tweet_id=int(row["customer_tweet_id"]),
            customer_author_id=row["customer_author_id"],
            customer_message=row["cleaned_customer_message"],
            cleaned_customer_message=row["cleaned_customer_message"],
            expected_intent=row["expected_intent"],
            expected_action=row["expected_action"],
            escalation_reason=row["escalation_reason"],
        )
        row["review_status"] = "ai_reviewed"
        row["reviewer_id"] = "cursor-ai-semantic-review-v1"
        row["reviewed_at"] = reviewed_at

    with open(REVIEW_QUEUE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Applied semantic reviews to {len(rows)} rows in {REVIEW_QUEUE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "apply-ai-reviews", "finalize"),
        help="Prepare, apply semantic review batches, or finalize reviewed rows.",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_review_queue()
    elif args.command == "apply-ai-reviews":
        apply_ai_reviews()
    else:
        finalize_golden_set()
