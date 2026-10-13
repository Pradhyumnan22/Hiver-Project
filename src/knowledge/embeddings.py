"""Dense embedding generation for semantic retrieval."""
from typing import List
import numpy as np
from src.utils.logging import setup_logger

logger = setup_logger("embeddings")


class DenseEmbedder:
    """Generates normalized vector embeddings for semantic search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval. "
                "Install the project requirements before building the index."
            ) from exc
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"SentenceTransformer loaded successfully (dim={self._dimension}).")

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode list of strings into L2-normalized float32 numpy array."""
        self._load_model()
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
