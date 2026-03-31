"""Embedding manager with pluggable backends: transformers (in-process) or llama-server (HTTP).

Both backends implement the same EmbeddingManager interface:
- encode(texts) -> np.ndarray of shape (n, dimension), L2-normalized
- serialize_embedding / deserialize_embedding for SQLite BLOB storage
- cosine_similarity for search
"""

import json
import logging
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Base embedding manager interface.

    Subclasses must implement encode(). Other methods have default implementations.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized embeddings.

        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32.
        """
        raise NotImplementedError

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Convert numpy embedding to bytes for SQLite BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    def deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert BLOB bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32).reshape(self.dimension)

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray, corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between a query and corpus of embeddings.

        Both inputs are assumed to be L2-normalized (as produced by encode()).

        Returns:
            Shape (n,) array of similarity scores in [-1, 1].
        """
        return corpus_embeddings @ query_embedding

    @staticmethod
    def estimate_variance(embedding: np.ndarray) -> np.ndarray:
        """Estimate per-dimension variance from signal magnitude.

        High-magnitude dimensions are assumed to be high-confidence (low variance).
        Based on Fisher information approach from SLM-V3 paper.

        Returns:
            Shape (d,) array of variance estimates with dtype float32.
        """
        sigma_min, sigma_max = 0.01, 1.0
        abs_emb = np.abs(embedding)
        max_val = np.maximum(abs_emb.max(), 1e-9)
        variance = sigma_max - (sigma_max - sigma_min) * (abs_emb / max_val)
        return variance.astype(np.float32) + 1e-9

    @staticmethod
    def fisher_rao_similarity(
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        corpus_variances: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Fisher-information-weighted similarity.

        Dimensions with low variance (high confidence) contribute more to the
        distance. This outperforms cosine similarity by +7.2pp on complex queries
        (SLM-V3 ablation study).

        Args:
            query_embedding: Shape (d,) query vector.
            corpus_embeddings: Shape (n, d) corpus matrix.
            corpus_variances: Shape (n, d) per-memory variance estimates.
            temperature: Scaling factor for the exponential.

        Returns:
            Shape (n,) array of similarity scores in [0, 1].
        """
        diff = corpus_embeddings - query_embedding  # (n, d)
        weighted_dist = np.sum(diff ** 2 / corpus_variances, axis=1)  # (n,)
        return np.exp(-weighted_dist / temperature)

    def serialize_variance(self, variance: np.ndarray) -> bytes:
        """Convert variance vector to bytes for SQLite BLOB storage."""
        return variance.astype(np.float32).tobytes()

    def deserialize_variance(self, blob: bytes) -> np.ndarray:
        """Convert BLOB bytes back to variance array."""
        return np.frombuffer(blob, dtype=np.float32).reshape(self.dimension)


class TransformersEmbeddingManager(EmbeddingManager):
    """Embedding manager using HuggingFace transformers (in-process).

    Default: sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim, ~80MB).
    Model is lazy-loaded on first encode() call.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        super().__init__(dimension=dimension)
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device: Optional[str] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading embedding model %s on %s", self.model_name, self._device
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Embedding model loaded successfully")

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._model(**inputs)

            # Mean pooling over token embeddings, masked by attention
            token_embs = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
            summed = torch.sum(token_embs * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = summed / counts

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)


class LlamaServerEmbeddingManager(EmbeddingManager):
    """Embedding manager using llama-server HTTP /embedding endpoint.

    Requires a running llama-server process:
        llama-server --model <gguf-path> --port 8787 --embedding --ctx-size 512

    The server keeps the model loaded persistently, so each request only pays
    inference cost (~7ms) rather than model load cost (~370ms).
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8787",
        dimension: int = 768,
        timeout: float = 30.0,
    ):
        super().__init__(dimension=dimension)
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._verified = False

    def _verify_server(self) -> None:
        """Check server is reachable on first use. Raises RuntimeError if not."""
        if self._verified:
            return
        try:
            req = urllib.request.Request(f"{self.server_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                health = json.loads(resp.read())
            if health.get("status") != "ok":
                raise RuntimeError(
                    f"llama-server at {self.server_url} returned unhealthy status: {health}"
                )
            self._verified = True
            logger.info("llama-server verified at %s", self.server_url)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to llama-server at {self.server_url}: {e}. "
                f"Start it with: llama-server --model <gguf> --port 8787 --embedding --ctx-size 512"
            ) from e

    def encode(self, texts: list[str]) -> np.ndarray:
        self._verify_server()

        embeddings = []
        for text in texts:
            payload = json.dumps({"content": text}).encode()
            req = urllib.request.Request(
                f"{self.server_url}/embedding",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())

            # Response format: [{index: 0, embedding: [[768 floats]]}]
            emb = data[0]["embedding"]
            if isinstance(emb[0], list):
                emb = emb[0]
            embeddings.append(emb)

        result = np.array(embeddings, dtype=np.float32)

        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        result = result / norms

        return result


def create_embedding_manager(
    backend: str = "transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dimension: int = 384,
    server_url: str = "http://localhost:8787",
) -> EmbeddingManager:
    """Factory function to create the appropriate embedding manager.

    Args:
        backend: 'transformers' for in-process HuggingFace model,
                 'llama-server' for external llama-server HTTP endpoint.
        model_name: Model name (used for transformers backend).
        dimension: Embedding dimension (384 for MiniLM, 768 for Gemma).
        server_url: URL of llama-server (used for llama-server backend).

    Returns:
        An EmbeddingManager instance.
    """
    if backend == "transformers":
        return TransformersEmbeddingManager(
            model_name=model_name, dimension=dimension
        )
    elif backend == "llama-server":
        return LlamaServerEmbeddingManager(
            server_url=server_url, dimension=dimension
        )
    else:
        raise ValueError(
            f"Unknown embedding backend: {backend!r}. "
            f"Use 'transformers' or 'llama-server'."
        )
