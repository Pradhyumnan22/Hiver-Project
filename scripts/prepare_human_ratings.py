"""Prepare a deterministic, judge-blind human review subset."""
import csv
import json
from collections import defaultdict
from pathlib import Path

AGENT_RESULTS_PATH = Path("reports/agent_results.json")
GOLDEN_PATH = Path("data/processed/golden_eval_set.json")
KNOWLEDGE_METADATA_PATH = Path("data/processed/knowledge_metadata.json")
PACKET_PATH = Path("data/evaluation/human_review_packet.json")
RATINGS_PATH = Path("data/evaluation/human_ratings.csv")
SUBSET_SIZE = 30


def main() -> None:
    with open(AGENT_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_by_id = {row["id"]: row for row in json.load(f)}
    with open(KNOWLEDGE_METADATA_PATH, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    evidence_by_id = {
        row["pair_id"]: {
            "source_id": row["pair_id"],
            "historical_customer_issue": row["cleaned_customer_text"],
            "historical_brand_reply": row["cleaned_brand_reply"],
        }
        for row in knowledge["records"]
    }

    # Only judge-rated auto-handled responses are eligible. Sort difficult
    # low-similarity cases first, then round-robin intents for diversity.
    buckets = defaultdict(list)
    for prediction in results["predictions"]:
        if prediction.get("reply") and prediction.get("judge"):
            buckets[prediction["predicted_intent"]].append(prediction)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: (
                row.get("retrieval_best_score") or 0.0,
                row["id"],
            )
        )

    selected = []
    intent_names = sorted(buckets)
    while len(selected) < SUBSET_SIZE:
        made_progress = False
        for intent in intent_names:
            if buckets[intent] and len(selected) < SUBSET_SIZE:
                selected.append(buckets[intent].pop(0))
                made_progress = True
        if not made_progress:
            break
    if len(selected) != SUBSET_SIZE:
        raise ValueError(f"Only {len(selected)} eligible responses were available.")

    packet = []
    for prediction in selected:
        golden = golden_by_id[prediction["id"]]
        packet.append(
            {
                "id": prediction["id"],
                "customer_message": golden["cleaned_customer_message"],
                "expected_intent": golden["expected_intent"],
                "predicted_intent": prediction["predicted_intent"],
                "reply_draft": prediction["reply"],
                "evidence": [
                    evidence_by_id[source_id]
                    for source_id in prediction["evidence_ids"]
                    if source_id in evidence_by_id
                ],
            }
        )

    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PACKET_PATH, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)

    with open(RATINGS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "groundedness",
                "relevance",
                "helpfulness",
                "consistency",
                "safety",
                "reviewer_id",
                "notes",
            ]
        )
        for item in packet:
            writer.writerow([item["id"], "", "", "", "", "", "", ""])


if __name__ == "__main__":
    main()
