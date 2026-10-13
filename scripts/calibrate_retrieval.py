"""Calibrate the dense-retrieval threshold on a development-only author split."""
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_golden_set import classify_candidate
from src.config import load_config
from src.knowledge.calibration import calibrate_threshold
from src.knowledge.retriever import FAISSRetriever, HistoricalRecord
from src.utils.logging import setup_logger

logger = setup_logger("calibrate_retrieval")
OUTPUT_PATH = Path("reports/retrieval_calibration.json")


def _to_record(pair: dict) -> HistoricalRecord:
    intent, _, _, _ = classify_candidate(
        pair["cleaned_customer_text"],
        pair["cleaned_brand_text"],
    )
    return HistoricalRecord(
        pair_id=pair["pair_id"],
        customer_tweet_id=pair["customer_tweet_id"],
        customer_author_id=pair["customer_author_id"],
        customer_text=pair["customer_text"],
        cleaned_customer_text=pair["cleaned_customer_text"],
        historical_brand_reply=pair["brand_text"],
        cleaned_brand_reply=pair["cleaned_brand_text"],
        intent=intent.value,
    )


def main() -> dict:
    config = load_config()
    dev_path = Path(config.data.processed_dir) / "dev_pairs.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_pairs = json.load(f)

    authors = sorted({str(pair["customer_author_id"]) for pair in dev_pairs})
    rng = random.Random(config.project.seed)
    rng.shuffle(authors)
    calibration_authors = set(authors[: max(1, int(len(authors) * 0.20))])

    index_records = [
        _to_record(pair)
        for pair in dev_pairs
        if str(pair["customer_author_id"]) not in calibration_authors
    ]
    calibration_records = [
        _to_record(pair)
        for pair in dev_pairs
        if str(pair["customer_author_id"]) in calibration_authors
    ]

    # Deduplicate calibration queries by the same fingerprint used by indexing.
    unique_calibration = {}
    for record in calibration_records:
        fingerprint = FAISSRetriever.text_fingerprint(record.cleaned_customer_text)
        unique_calibration.setdefault(fingerprint, record)
    calibration_records = list(unique_calibration.values())[:500]
    calibration_texts = {
        record.cleaned_customer_text for record in calibration_records
    }

    retriever = FAISSRetriever(relevance_threshold=0.0)
    retriever.build_index(index_records, exclude_texts=calibration_texts)
    query_embeddings = retriever.embedder.encode(
        [record.cleaned_customer_text for record in calibration_records]
    )
    scores_array, indices_array = retriever.index.search(query_embeddings, 1)

    scores = [float(row[0]) for row in scores_array]
    relevant = [
        retriever.records[int(index_row[0])].intent == query.intent
        for query, index_row in zip(calibration_records, indices_array)
    ]
    result = calibrate_threshold(
        scores,
        relevant,
        candidates=[value / 100 for value in range(20, 76, 5)],
        target_precision=0.80,
        minimum_coverage=0.10,
    )

    report = {
        "method": "development-only author split; top-1 same-intent proxy",
        "seed": config.project.seed,
        "indexed_records": len(retriever.records),
        "calibration_queries": len(calibration_records),
        "target_proxy_precision": result.target_precision,
        "minimum_coverage": result.minimum_coverage,
        "selected_threshold": result.selected_threshold,
        "limitations": (
            "Same weakly supervised intent is a relevance proxy, not a human "
            "judgement. Final retrieval quality requires manual or judge review."
        ),
        "threshold_metrics": [asdict(metric) for metric in result.metrics],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"Selected threshold {result.selected_threshold:.2f} using "
        f"{len(calibration_records)} development-only queries."
    )
    return report


if __name__ == "__main__":
    main()
