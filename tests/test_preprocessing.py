"""Tests for data preprocessing, text cleaning, conversation linking, and leakage prevention."""
import pytest
from src.data.cleaner import TextCleaner
from src.data.schemas import ConversationPair
from src.data.loader import DataLoader


def test_text_cleaner_masks_pii_urls_and_handles():
    raw_text = "@AmazonHelp My order 123-456 did not arrive! Email me at test@example.com or call 555-123-4567. Link: https://amazon.com/track"
    cleaned = TextCleaner.clean(raw_text)

    assert "@user" in cleaned
    assert "<URL>" in cleaned
    assert "<EMAIL>" in cleaned
    assert "<PHONE>" in cleaned
    assert "https://amazon.com" not in cleaned
    assert "test@example.com" not in cleaned
    assert "555-123-4567" not in cleaned


def test_text_cleaner_english_filter():
    english_text = "Where is my package? It was supposed to be delivered yesterday."
    non_english_text = "こんにちは、荷物が届いていません。確認してください。"

    assert TextCleaner.is_english_text(english_text) is True
    assert TextCleaner.is_english_text(non_english_text) is False


def test_partition_dataset_guarantees_zero_leakage():
    # Construct mock pairs across 4 distinct customer authors
    mock_pairs = [
        ConversationPair(
            pair_id="p1", customer_tweet_id=1, customer_author_id="user_A", customer_text="Issue 1",
            cleaned_customer_text="Issue 1", brand_tweet_id=10, brand_author_id="AmazonHelp",
            brand_text="Reply 1", cleaned_brand_text="Reply 1"
        ),
        ConversationPair(
            pair_id="p2", customer_tweet_id=2, customer_author_id="user_A", customer_text="Issue 2",
            cleaned_customer_text="Issue 2", brand_tweet_id=11, brand_author_id="AmazonHelp",
            brand_text="Reply 2", cleaned_brand_text="Reply 2"
        ),
        ConversationPair(
            pair_id="p3", customer_tweet_id=3, customer_author_id="user_B", customer_text="Issue 3",
            cleaned_customer_text="Issue 3", brand_tweet_id=12, brand_author_id="AmazonHelp",
            brand_text="Reply 3", cleaned_brand_text="Reply 3"
        ),
        ConversationPair(
            pair_id="p4", customer_tweet_id=4, customer_author_id="user_C", customer_text="Issue 4",
            cleaned_customer_text="Issue 4", brand_tweet_id=13, brand_author_id="AmazonHelp",
            brand_text="Reply 4", cleaned_brand_text="Reply 4"
        ),
        ConversationPair(
            pair_id="p5", customer_tweet_id=5, customer_author_id="user_D", customer_text="Issue 5",
            cleaned_customer_text="Issue 5", brand_tweet_id=14, brand_author_id="AmazonHelp",
            brand_text="Reply 5", cleaned_brand_text="Reply 5"
        ),
    ]

    loader = DataLoader(raw_csv_path="dummy.csv", brand="AmazonHelp", dev_ratio=0.5, seed=42)
    dev_pairs, eval_pairs = loader.partition_dataset(mock_pairs)

    dev_authors = {p.customer_author_id for p in dev_pairs}
    eval_authors = {p.customer_author_id for p in eval_pairs}
    dev_pair_ids = {p.pair_id for p in dev_pairs}
    eval_pair_ids = {p.pair_id for p in eval_pairs}

    # Strict zero-leakage assertions
    assert len(dev_authors.intersection(eval_authors)) == 0, "Found customer author overlap between dev and eval!"
    assert len(dev_pair_ids.intersection(eval_pair_ids)) == 0, "Found pair ID overlap between dev and eval!"
    assert len(dev_pairs) + len(eval_pairs) == len(mock_pairs)
