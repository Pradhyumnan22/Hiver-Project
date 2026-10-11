"""Script to construct, annotate, and strictly validate the 200 golden evaluation examples from held-out eval partition."""
import json
import re
from pathlib import Path
from typing import List
from src.data.intent_taxonomy import Action, EscalationReason, GoldenEvaluationExample, Intent
from src.utils.logging import setup_logger

logger = setup_logger("build_golden_set")

EVAL_PAIRS_PATH = Path("data/processed/eval_pairs.json")
OUTPUT_GOLDEN_PATH = Path("data/processed/golden_eval_set.json")


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


def build_golden_set():
    logger.info(f"Loading held-out evaluation pairs from {EVAL_PAIRS_PATH}")
    with open(EVAL_PAIRS_PATH, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)

    # Stratify across intents
    buckets = {intent: [] for intent in Intent}

    for p in eval_pairs:
        text = p["cleaned_customer_text"]
        brand_reply = p["cleaned_brand_text"]
        intent, action, reason, notes = classify_candidate(text, brand_reply)

        example = GoldenEvaluationExample(
            id=f"gold_{p['customer_tweet_id']}",
            customer_tweet_id=p["customer_tweet_id"],
            customer_author_id=p["customer_author_id"],
            customer_message=p["customer_text"],
            cleaned_customer_message=p["cleaned_customer_text"],
            conversation_context=f"Customer reached out to @AmazonHelp via Twitter.",
            expected_intent=intent,
            expected_action=action,
            escalation_reason=reason,
            annotation_notes=notes,
            historical_brand_reply=p["cleaned_brand_text"],
            is_synthetic=False,
        )
        buckets[intent].append(example)

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

    selected_examples: List[GoldenEvaluationExample] = []
    for intent, count in allocations.items():
        available = buckets[intent]
        if len(available) < count:
            logger.warning(f"Intent {intent.value} has only {len(available)} candidates; taking all.")
            selected_examples.extend(available)
        else:
            selected_examples.extend(available[:count])

    logger.info(f"Total golden examples assembled: {len(selected_examples)}")

    # Verify every example passes Pydantic model validation
    validated_data = [ex.model_dump() for ex in selected_examples]

    OUTPUT_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(validated_data, f, indent=2)

    logger.info(f"Saved verified golden evaluation set to {OUTPUT_GOLDEN_PATH}")


if __name__ == "__main__":
    build_golden_set()
