"""Tests for FAISS historical retriever, leakage prevention, and relevance thresholding."""
import hashlib
import numpy as np
import pytest
from src.knowledge.retriever import FAISSRetriever, HistoricalRecord


class DeterministicTestEmbedder:
    """Small credential-free embedder used only by unit tests."""

    dimension = 64

    def encode(self, texts, batch_size=64):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:2], "big") % self.dimension] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


@pytest.fixture
def test_embedder():
    return DeterministicTestEmbedder()


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


def test_retriever_build_and_leakage_exclusion(sample_records, test_embedder):
    retriever = FAISSRetriever(embedder=test_embedder, relevance_threshold=0.40)
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


def test_retriever_query_and_relevance_threshold(sample_records, test_embedder):
    retriever = FAISSRetriever(embedder=test_embedder, relevance_threshold=0.20)
    retriever.build_index(sample_records[:2])

    # Query matching delivery
    res = retriever.retrieve("Where is my delivery? Package delayed", top_k=2)
    assert not res.is_empty
    assert len(res.records) >= 1
    assert res.scores[0] >= 0.20
    assert "rec_1" in res.source_ids



def test_retriever_empty_result_on_low_similarity(sample_records, test_embedder):
    # Set high threshold
    retriever = FAISSRetriever(embedder=test_embedder, relevance_threshold=0.99)
    retriever.build_index(sample_records[:2])

    # Irrelevant query
    res = retriever.retrieve("quantum physics string theory simulation", top_k=2)
    assert res.is_empty
    assert len(res.records) == 0
    assert len(res.scores) == 0
    assert len(res.source_ids) == 0


def test_retriever_filters_duplicate_and_excluded_text(sample_records, test_embedder):
    duplicate = sample_records[0].model_copy(
        update={
            "pair_id": "duplicate",
            "customer_tweet_id": 999,
            "customer_author_id": "another_author",
        }
    )
    retriever = FAISSRetriever(embedder=test_embedder)
    retriever.build_index(
        [*sample_records[:2], duplicate],
        exclude_texts={sample_records[1].cleaned_customer_text},
    )

    assert [record.pair_id for record in retriever.records] == ["rec_1"]


def test_retriever_save_load_round_trip(
    sample_records,
    test_embedder,
    tmp_path,
):
    index_path = tmp_path / "knowledge.index"
    metadata_path = tmp_path / "knowledge.json"
    original = FAISSRetriever(embedder=test_embedder, relevance_threshold=0.2)
    original.build_index(sample_records[:2])
    original.save(index_path, metadata_path)

    loaded = FAISSRetriever(embedder=test_embedder).load(
        index_path,
        metadata_path,
    )
    result = loaded.retrieve("Where is my delayed delivery?", top_k=1)

    assert loaded.relevance_threshold == 0.2
    assert result.source_ids == ["rec_1"]
