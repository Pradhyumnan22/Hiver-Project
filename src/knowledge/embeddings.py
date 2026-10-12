"""Dense embedding generator with graceful offline fallback."""
from typing import List, Optional
import numpy as np
from src.utils.logging import setup_logger

logger = setup_logger("embeddings")


class DenseEmbedder:
    """Generates normalized vector embeddings for semantic search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_vectorizer = None
        self._dimension = 384

    def _load_model(self):
        if self._model is not None or self._fallback_vectorizer is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer loaded successfully (dim={self._dimension}).")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to fast normalized TF-IDF dense projection.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._fallback_vectorizer = TfidfVectorizer(max_features=self._dimension, sublinear_tf=True)

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode list of strings into L2-normalized float32 numpy array."""
        self._load_model()
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        else:
            # Fallback path for offline/unit tests
            if not hasattr(self._fallback_vectorizer, "vocabulary_") or self._fallback_vectorizer.vocabulary_ is None:
                self._fallback_vectorizer.fit(texts + ["package delivery refund return item account"])
            sparse = self._fallback_vectorizer.transform(texts).toarray()
            # Normalize L2
            norms = np.linalg.norm(sparse, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense = (sparse / norms).astype(np.float32)
            # Pad to dimension if needed
            if dense.shape[1] < self._dimension:
                pad = np.zeros((dense.shape[0], self._dimension - dense.shape[1]), dtype=np.float32)
                dense = np.hstack([dense, pad])
            return dense
