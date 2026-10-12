"""Baseline classifiers and reply generation models."""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from src.data.intent_taxonomy import Action, Intent
from src.utils.logging import setup_logger

logger = setup_logger("baselines")


class MajorityClassBaseline:
    """Trivial baseline predicting the most frequent class observed during training."""

    def __init__(self):
        self.majority_class: Optional[str] = None
        self.class_counts: Dict[str, int] = {}

    def fit(self, texts: List[str], labels: List[str]) -> "MajorityClassBaseline":
        counter = Counter(labels)
        self.class_counts = dict(counter)
        self.majority_class = counter.most_common(1)[0][0]
        logger.info(f"MajorityClassBaseline fit. Majority class: {self.majority_class}")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self.majority_class:
            raise ValueError("Model has not been fitted yet.")
        return [self.majority_class] * len(texts)


class TfidfLogisticRegressionBaseline:
    """TF-IDF n-gram vectorizer paired with a multinomial Logistic Regression classifier."""

    def __init__(self, max_features: int = 5000, c_param: float = 1.0, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            C=c_param,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        self.classes_: Optional[List[str]] = None

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfLogisticRegressionBaseline":
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)
        logger.info(f"TfidfLogisticRegression fitted on {len(texts)} samples across {len(self.classes_)} classes.")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        preds = self.classifier.predict(X)
        return list(preds)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)


class SimpleRetrievalReplyBaseline:
    """Simple baseline retrieving the nearest historical reply via TF-IDF cosine similarity."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.historical_queries: List[str] = []
        self.historical_replies: List[str] = []
        self.query_vectors = None

    def fit(self, customer_texts: List[str], brand_replies: List[str]) -> "SimpleRetrievalReplyBaseline":
        self.historical_queries = customer_texts
        self.historical_replies = brand_replies
        self.query_vectors = self.vectorizer.fit_transform(customer_texts)
        logger.info(f"SimpleRetrievalReplyBaseline indexed {len(customer_texts)} historical pairs.")
        return self

    def reply(self, query: str) -> Tuple[str, float]:
        """Retrieve most similar historical reply and cosine similarity score."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.query_vectors)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return self.historical_replies[best_idx], round(best_score, 4)
