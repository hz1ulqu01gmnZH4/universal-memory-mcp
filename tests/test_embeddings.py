"""Tests for embeddings.py — factory, backends, serialization."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest

from embeddings import (
    EmbeddingManager,
    LlamaServerEmbeddingManager,
    TransformersEmbeddingManager,
    create_embedding_manager,
)


class TestFactory:
    def test_create_transformers_backend(self):
        mgr = create_embedding_manager(
            backend="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
        )
        assert isinstance(mgr, TransformersEmbeddingManager)
        assert mgr.dimension == 384

    def test_create_llama_server_backend(self):
        mgr = create_embedding_manager(
            backend="llama-server",
            dimension=768,
            server_url="http://localhost:9999",
        )
        assert isinstance(mgr, LlamaServerEmbeddingManager)
        assert mgr.dimension == 768
        assert mgr.server_url == "http://localhost:9999"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            create_embedding_manager(backend="invalid")


class TestBaseManager:
    def test_serialize_deserialize(self):
        mgr = create_embedding_manager(
            backend="llama-server", dimension=4, server_url="http://unused:0"
        )
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        blob = mgr.serialize_embedding(emb)
        restored = mgr.deserialize_embedding(blob)
        np.testing.assert_array_almost_equal(emb, restored)

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        sims = EmbeddingManager.cosine_similarity(a, b)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)

    def test_base_encode_raises(self):
        mgr = EmbeddingManager(dimension=384)
        with pytest.raises(NotImplementedError):
            mgr.encode(["test"])


# --- Mock HTTP server for llama-server tests ---


def _make_mock_handler(dim: int):
    """Create a handler that returns fake embeddings of given dimension."""

    class MockLlamaHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/embedding":
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len))
                # Generate a deterministic fake embedding based on content hash
                seed = hash(body.get("content", "")) % (2**31)
                rng = np.random.RandomState(seed)
                emb = rng.randn(dim).astype(float).tolist()
                response = [{"index": 0, "embedding": [emb]}]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress request logging in tests

    return MockLlamaHandler


@pytest.fixture
def mock_llama_server():
    """Start a mock llama-server on a random port."""
    handler = _make_mock_handler(dim=768)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestVarianceAndFisherRao:
    def test_estimate_variance_shape(self):
        emb = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)
        var = EmbeddingManager.estimate_variance(emb)
        assert var.shape == emb.shape
        assert var.dtype == np.float32

    def test_estimate_variance_high_magnitude_low_variance(self):
        emb = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        var = EmbeddingManager.estimate_variance(emb)
        # Highest magnitude dimension should have lowest variance
        assert var[0] < var[2] < var[1]

    def test_estimate_variance_all_positive(self):
        emb = np.random.randn(384).astype(np.float32)
        var = EmbeddingManager.estimate_variance(emb)
        assert (var > 0).all()

    def test_fisher_rao_similarity_identical(self):
        q = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        corpus = q.reshape(1, -1)
        var = np.ones((1, 3), dtype=np.float32) * 0.5
        sims = EmbeddingManager.fisher_rao_similarity(q, corpus, var)
        assert sims[0] == pytest.approx(1.0)

    def test_fisher_rao_similarity_distant(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        corpus = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        var = np.ones((1, 3), dtype=np.float32) * 0.1
        sims = EmbeddingManager.fisher_rao_similarity(q, corpus, var)
        assert sims[0] < 0.01  # very distant with low variance

    def test_fisher_rao_low_variance_more_discriminative(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.array([[0.9, 0.1]], dtype=np.float32)
        # Low variance = more discriminative (smaller score for same distance)
        low_var = np.ones((1, 2), dtype=np.float32) * 0.01
        high_var = np.ones((1, 2), dtype=np.float32) * 1.0
        sim_low = EmbeddingManager.fisher_rao_similarity(q, corpus, low_var)[0]
        sim_high = EmbeddingManager.fisher_rao_similarity(q, corpus, high_var)[0]
        assert sim_low < sim_high

    def test_variance_serialize_roundtrip(self):
        mgr = EmbeddingManager(dimension=4)
        var = np.array([0.1, 0.5, 0.3, 0.9], dtype=np.float32)
        blob = mgr.serialize_variance(var)
        restored = mgr.deserialize_variance(blob)
        np.testing.assert_array_almost_equal(var, restored)

    def test_fisher_rao_empty_corpus(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.empty((0, 2), dtype=np.float32)
        var = np.empty((0, 2), dtype=np.float32)
        sims = EmbeddingManager.fisher_rao_similarity(q, corpus, var)
        assert len(sims) == 0


class TestLlamaServerBackend:
    def test_encode_returns_normalized(self, mock_llama_server):
        mgr = LlamaServerEmbeddingManager(
            server_url=mock_llama_server, dimension=768
        )
        result = mgr.encode(["hello world"])
        assert result.shape == (1, 768)
        assert result.dtype == np.float32
        # Check L2 normalized
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_encode_multiple_texts(self, mock_llama_server):
        mgr = LlamaServerEmbeddingManager(
            server_url=mock_llama_server, dimension=768
        )
        result = mgr.encode(["text one", "text two", "text three"])
        assert result.shape == (3, 768)

    def test_encode_deterministic(self, mock_llama_server):
        mgr = LlamaServerEmbeddingManager(
            server_url=mock_llama_server, dimension=768
        )
        r1 = mgr.encode(["same text"])
        r2 = mgr.encode(["same text"])
        np.testing.assert_array_equal(r1, r2)

    def test_different_texts_different_embeddings(self, mock_llama_server):
        mgr = LlamaServerEmbeddingManager(
            server_url=mock_llama_server, dimension=768
        )
        r1 = mgr.encode(["text A"])
        r2 = mgr.encode(["text B"])
        assert not np.array_equal(r1, r2)

    def test_server_not_running_raises(self):
        mgr = LlamaServerEmbeddingManager(
            server_url="http://127.0.0.1:1", dimension=768
        )
        with pytest.raises(RuntimeError, match="Cannot connect to llama-server"):
            mgr.encode(["test"])

    def test_serialize_roundtrip(self, mock_llama_server):
        mgr = LlamaServerEmbeddingManager(
            server_url=mock_llama_server, dimension=768
        )
        emb = mgr.encode(["roundtrip test"])[0]
        blob = mgr.serialize_embedding(emb)
        restored = mgr.deserialize_embedding(blob)
        np.testing.assert_array_almost_equal(emb, restored)
