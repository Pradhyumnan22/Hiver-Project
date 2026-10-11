"""Dataset loader, conversation pair linking, and leakage-safe partitioning."""
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple
import pandas as pd
from src.data.cleaner import TextCleaner
from src.data.schemas import ConversationPair, DatasetPartition
from src.utils.logging import setup_logger

logger = setup_logger("data_loader")


class DataLoader:
    """Streams twcs.csv, extracts brand-specific conversation pairs, and partitions without leakage."""

    def __init__(
        self,
        raw_csv_path: Path | str,
        brand: str = "AmazonHelp",
        dev_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.brand = brand
        self.dev_ratio = dev_ratio
        self.seed = seed

    def extract_pairs(
        self,
        max_rows: Optional[int] = 100000,
        chunksize: int = 25000,
        min_text_len: int = 15,
    ) -> List[ConversationPair]:
        """Extract customer-brand conversation pairs for the target brand."""
        logger.info(f"Extracting conversation pairs for brand '{self.brand}' from {self.raw_csv_path}")

        customer_tweets: Dict[int, Dict] = {}
        brand_tweets: List[Dict] = []

        rows_read = 0
        reader = pd.read_csv(self.raw_csv_path, chunksize=chunksize, dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "float64",
        })

        for chunk in reader:
            rows_read += len(chunk)

            # Store inbound customer tweets
            inbound = chunk[chunk["inbound"] == True]
            for _, row in inbound.iterrows():
                customer_tweets[int(row["tweet_id"])] = {
                    "tweet_id": int(row["tweet_id"]),
                    "author_id": str(row["author_id"]),
                    "text": str(row["text"]),
                    "created_at": str(row["created_at"]),
                }

            # Store brand responses that respond to an inbound tweet
            brand_resp = chunk[
                (chunk["author_id"] == self.brand)
                & (chunk["inbound"] == False)
                & (chunk["in_response_to_tweet_id"].notna())
            ]
            for _, row in brand_resp.iterrows():
                brand_tweets.append({
                    "tweet_id": int(row["tweet_id"]),
                    "author_id": str(row["author_id"]),
                    "text": str(row["text"]),
                    "created_at": str(row["created_at"]),
                    "in_response_to_tweet_id": int(row["in_response_to_tweet_id"]),
                })

            if max_rows and rows_read >= max_rows:
                logger.info(f"Reached max_rows limit ({max_rows}). Stopping stream.")
                break

        # Match brand response to parent customer tweet
        matched_pairs: List[ConversationPair] = []
        for brand_t in brand_tweets:
            parent_id = brand_t["in_response_to_tweet_id"]
            if parent_id in customer_tweets:
                cust_t = customer_tweets[parent_id]

                # Filter English only
                if not (TextCleaner.is_english_text(cust_t["text"]) and TextCleaner.is_english_text(brand_t["text"])):
                    continue

                cleaned_cust = TextCleaner.clean(cust_t["text"])
                cleaned_brand = TextCleaner.clean(brand_t["text"])

                # Skip trivial noise or ultra-short tweets
                if len(cleaned_cust) < min_text_len or len(cleaned_brand) < min_text_len:
                    continue

                pair = ConversationPair(
                    pair_id=f"{self.brand}_{brand_t['tweet_id']}_{cust_t['tweet_id']}",
                    customer_tweet_id=cust_t["tweet_id"],
                    customer_author_id=cust_t["author_id"],
                    customer_text=cust_t["text"],
                    cleaned_customer_text=cleaned_cust,
                    brand_tweet_id=brand_t["tweet_id"],
                    brand_author_id=brand_t["author_id"],
                    brand_text=brand_t["text"],
                    cleaned_brand_text=cleaned_brand,
                    created_at_customer=cust_t["created_at"],
                    created_at_brand=brand_t["created_at"],
                )
                matched_pairs.append(pair)

        logger.info(f"Total matched high-quality conversation pairs: {len(matched_pairs)}")
        return matched_pairs

    def partition_dataset(
        self,
        pairs: List[ConversationPair],
    ) -> Tuple[List[ConversationPair], List[ConversationPair]]:
        """Split pairs into dev and eval sets by customer_author_id to guarantee zero leakage."""
        import random
        from datetime import datetime

        # Group conversation pairs by customer author_id
        author_to_pairs: Dict[str, List[ConversationPair]] = {}
        for p in pairs:
            author_to_pairs.setdefault(p.customer_author_id, []).append(p)

        authors = sorted(list(author_to_pairs.keys()))
        rng = random.Random(self.seed)
        rng.shuffle(authors)

        split_idx = int(len(authors) * self.dev_ratio)
        dev_authors = set(authors[:split_idx])
        eval_authors = set(authors[split_idx:])

        dev_pairs: List[ConversationPair] = []
        eval_pairs: List[ConversationPair] = []

        for p in pairs:
            if p.customer_author_id in dev_authors:
                dev_pairs.append(p)
            else:
                eval_pairs.append(p)

        logger.info(
            f"Dataset partitioned: {len(dev_pairs)} dev pairs ({len(dev_authors)} authors), "
            f"{len(eval_pairs)} eval pairs ({len(eval_authors)} authors). Zero author/conversation overlap."
        )
        return dev_pairs, eval_pairs
