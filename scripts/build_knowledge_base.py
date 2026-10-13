"""Script to build and persist the FAISS knowledge base index from development pairs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from scripts.build_golden_set import classify_candidate
from src.knowledge.retriever import FAISSRetriever, HistoricalRecord
from src.utils.logging import setup_logger

logger = setup_logger("build_knowledge_base")


def main():
    config = load_config()
    dev_path = Path(config.data.processed_dir) / "dev_pairs.json"
    eval_path = Path(config.data.processed_dir) / "eval_pairs.json"
    golden_path = Path(config.data.golden_set_path)

    logger.info(f"Loading development conversation pairs from {dev_path}")
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_pairs = json.load(f)

    # Collect all evaluation and golden authors and tweets for exclusion
    excluded_authors = set()
    excluded_tweets = set()
    excluded_texts = set()

    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_pairs = json.load(f)
        for ep in eval_pairs:
            excluded_authors.add(ep["customer_author_id"])
            excluded_tweets.add(ep["customer_tweet_id"])
            excluded_texts.add(ep["cleaned_customer_text"])

    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_pairs = json.load(f)
        for gp in golden_pairs:
            excluded_authors.add(gp["customer_author_id"])
            excluded_tweets.add(gp["customer_tweet_id"])
            excluded_texts.add(gp["cleaned_customer_message"])

    logger.info(
        f"Strict leakage exclusion list prepared: {len(excluded_authors)} authors, {len(excluded_tweets)} tweet IDs."
    )

    records = []
    for p in dev_pairs:
        intent, _, _, _ = classify_candidate(
            p["cleaned_customer_text"],
            p["cleaned_brand_text"],
        )
        records.append(HistoricalRecord(
            pair_id=p["pair_id"],
            customer_tweet_id=p["customer_tweet_id"],
            customer_author_id=p["customer_author_id"],
            customer_text=p["customer_text"],
            cleaned_customer_text=p["cleaned_customer_text"],
            historical_brand_reply=p["brand_text"],
            cleaned_brand_reply=p["cleaned_brand_text"],
            intent=intent.value,
        ))

    retriever = FAISSRetriever(relevance_threshold=config.retrieval.relevance_threshold)
    retriever.build_index(
        records,
        exclude_author_ids=excluded_authors,
        exclude_tweet_ids=excluded_tweets,
        exclude_texts=excluded_texts,
    )

    index_path = Path(config.retrieval.faiss_index_path)
    metadata_path = Path(config.retrieval.metadata_path)
    retriever.save(index_path, metadata_path)
    logger.info(f"Knowledge base ready! Indexed {len(retriever.records)} items.")


if __name__ == "__main__":
    main()
