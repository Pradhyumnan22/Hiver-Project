"""FAISS-based historical conversation retriever with strict leakage exclusion and relevance thresholding."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field
from src.knowledge.embeddings import DenseEmbedder
from src.utils.logging import setup_logger

logger = setup_logger("faiss_retriever")


class HistoricalRecord(BaseModel):
    """Schema for an individual indexed historical customer support pair."""
    pair_id: str
    customer_tweet_id: int
    customer_author_id: str
    customer_text: str
    cleaned_customer_text: str
    historical_brand_reply: str
    cleaned_brand_reply: str
    intent: Optional[str] = None


class RetrievalResult(BaseModel):
    """Structured output returned by retriever for an incoming query."""
    query: str
    records: List[HistoricalRecord]
    scores: List[float]
    source_ids: List[str]
    is_empty: bool
    relevance_threshold: float
    best_score: float = 0.0


class FAISSRetriever:
    """Retriever indexing customer support cases using FAISS cosine similarity (IndexFlatIP)."""

    def __init__(
        self,
        embedder: Optional[DenseEmbedder] = None,
        relevance_threshold: float = 0.45,
    ):
        self.embedder = embedder or DenseEmbedder()
        self.relevance_threshold = relevance_threshold
        self.records: List[HistoricalRecord] = []
        self.index = None
        self._excluded_author_ids: Set[str] = set()
        self._excluded_tweet_ids: Set[int] = set()

    def build_index(
        self,
        records: List[HistoricalRecord],
        exclude_author_ids: Optional[Set[str]] = None,
        exclude_tweet_ids: Optional[Set[int]] = None,
    ) -> "FAISSRetriever":
        """Build FAISS index while strictly filtering out any held-out evaluation authors or tweets."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Please install faiss-cpu.")

        self._excluded_author_ids = exclude_author_ids or set()
        self._excluded_tweet_ids = exclude_tweet_ids or set()

        filtered_records = []
        filtered_texts = []

        for rec in records:
            # Strict leakage exclusion check
            if rec.customer_author_id in self._excluded_author_ids:
                continue
            if rec.customer_tweet_id in self._excluded_tweet_ids:
                continue

            filtered_records.append(rec)
            filtered_texts.append(rec.cleaned_customer_text)

        logger.info(
            f"Building FAISS index: {len(filtered_records)} indexed records "
            f"(filtered out {len(records) - len(filtered_records)} evaluation/leakage records)."
        )

        self.records = filtered_records
        if not self.records:
            logger.warning("No records to index!")
            return self

        # Generate embeddings
        embeddings = self.embedder.encode(filtered_texts)
        dimension = embeddings.shape[1]

        # Inner Product on normalized vectors equals exact Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {dimension}.")
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve top-k historical records exceeding relevance threshold."""
        thresh = threshold if threshold is not None else self.relevance_threshold

        if self.index is None or len(self.records) == 0:
            return RetrievalResult(
                query=query,
                records=[],
                scores=[],
                source_ids=[],
                is_empty=True,
                relevance_threshold=thresh,
                best_score=0.0,
            )

        q_emb = self.embedder.encode([query])
        scores_arr, indices_arr = self.index.search(q_emb, min(top_k * 2, len(self.records)))

        retrieved_records: List[HistoricalRecord] = []
        retrieved_scores: List[float] = []
        source_ids: List[str] = []

        best_score = float(scores_arr[0][0]) if len(scores_arr[0]) > 0 else 0.0

        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            score_val = round(float(score), 4)
            # Relevance threshold enforcement
            if score_val >= thresh:
                rec = self.records[idx]
                retrieved_records.append(rec)
                retrieved_scores.append(score_val)
                source_ids.append(rec.pair_id)

                if len(retrieved_records) >= top_k:
                    break

        is_empty = len(retrieved_records) == 0

        return RetrievalResult(
            query=query,
            records=retrieved_records,
            scores=retrieved_scores,
            source_ids=source_ids,
            is_empty=is_empty,
            relevance_threshold=thresh,
            best_score=round(best_score, 4),
        )

    def save(self, index_path: Path | str, metadata_path: Path | str):
        """Persist FAISS index binary and metadata records."""
        import faiss

        index_p = Path(index_path)
        meta_p = Path(metadata_path)
        index_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_p))
        meta_data = {
            "records": [r.model_dump() for r in self.records],
            "relevance_threshold": self.relevance_threshold,
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
        logger.info(f"Saved FAISS index ({index_p}) and metadata ({meta_p}).")

    def load(self, index_path: Path | str, metadata_path: Path | str) -> "FAISSRetriever":
        """Load persisted FAISS index and metadata records."""
        import faiss

        index_p = Path(index_path)
        meta_p = Path(metadata_path)
        if not (index_p.exists() and meta_p.exists()):
            raise FileNotFoundError(f"Missing index ({index_p}) or metadata ({meta_p})")

        self.index = faiss.read_index(str(index_p))
        with open(meta_p, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.records = [HistoricalRecord(**r) for r in meta_data["records"]]
        self.relevance_threshold = meta_data.get("relevance_threshold", 0.45)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} items from {index_p}.")
        return self
