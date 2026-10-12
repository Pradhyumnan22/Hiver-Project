"""Tests for FAISS historical retriever, leakage prevention, and relevance thresholding."""
import pytest
from src.knowledge.embeddings import DenseEmbedder
from src.knowledge.retriever import FAISSRetriever, HistoricalRecord


@pytest.fixture
def sample_records():
    return [
        HistoricalRecord(
            pair_id="rec_1",
            customer_tweet_id=101,
            customer_author_id="user_dev_1",
            customer_text="Where is my delivery? It has been 4 days.",
            cleaned_customer_text="Where is my delivery? It has been 4 days.",
            historical_brand_reply="Please track your package via your orders page.",
            cleaned_brand_reply="Please track your package via your orders page.",
            intent="order_status_delivery",
        ),
        HistoricalRecord(
            pair_id="rec_2",
            customer_tweet_id=102,
            customer_author_id="user_dev_2",
            customer_text="I want to return this broken coffee maker for a refund.",
            cleaned_customer_text="I want to return this broken coffee maker for a refund.",
            historical_brand_reply="You can start your return and refund in the Returns Center.",
            cleaned_brand_reply="You can start your return and refund in the Returns Center.",
            intent="returns_refunds",
        ),
        HistoricalRecord(
            pair_id="rec_3",
            customer_tweet_id=103,
            customer_author_id="user_eval_leaked",
            customer_text="Leaked evaluation inquiry that must not be indexed.",
            cleaned_customer_text="Leaked evaluation inquiry that must not be indexed.",
            historical_brand_reply="Leaked brand reply.",
            cleaned_brand_reply="Leaked brand reply.",
            intent="other_unknown",
        ),
    ]


def test_retriever_build_and_leakage_exclusion(sample_records):
    retriever = FAISSRetriever(relevance_threshold=0.40)
    # Exclude user_eval_leaked
    retriever.build_index(
        sample_records,
        exclude_author_ids={"user_eval_leaked"},
        exclude_tweet_ids={103},
    )

    # Assert leaked record was strictly excluded
    assert len(retriever.records) == 2
    indexed_ids = [r.pair_id for r in retriever.records]
    assert "rec_1" in indexed_ids
    assert "rec_2" in indexed_ids
    assert "rec_3" not in indexed_ids


def test_retriever_query_and_relevance_threshold(sample_records):
    retriever = FAISSRetriever(relevance_threshold=0.20)
    retriever.build_index(sample_records[:2])

    # Query matching delivery
    res = retriever.retrieve("Where is my delivery? Package delayed", top_k=2)
    assert not res.is_empty
    assert len(res.records) >= 1
    assert res.scores[0] >= 0.20
    assert "rec_1" in res.source_ids



def test_retriever_empty_result_on_low_similarity(sample_records):
    # Set high threshold
    retriever = FAISSRetriever(relevance_threshold=0.99)
    retriever.build_index(sample_records[:2])

    # Irrelevant query
    res = retriever.retrieve("quantum physics string theory simulation", top_k=2)
    assert res.is_empty
    assert len(res.records) == 0
    assert len(res.scores) == 0
    assert len(res.source_ids) == 0
