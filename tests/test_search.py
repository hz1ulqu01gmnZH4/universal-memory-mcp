"""Tests for search.py — FTS5 keyword search and hybrid search."""

import pytest

from memory_store import MemoryStore
from search import HybridSearch


@pytest.fixture
def populated_store(tmp_path):
    """Store with test memories for search testing."""
    db_path = str(tmp_path / "test.db")
    s = MemoryStore(db_path=db_path, embedding_manager=None)

    s.store_memory(
        content="Transformers use O(n^2) attention complexity for sequence processing",
        memory_type="semantic",
        agent_id="researcher",
        importance=0.9,
        compute_embedding=False,
    )
    s.store_memory(
        content="The experiment on GPU cluster started at 14:00 UTC",
        memory_type="episodic",
        agent_id="lab_manager",
        importance=0.7,
        compute_embedding=False,
    )
    s.store_memory(
        content="When encountering CUDA out of memory, reduce batch size first",
        memory_type="procedural",
        agent_id="researcher",
        importance=0.8,
        compute_embedding=False,
    )
    s.store_memory(
        content="BitNet uses 1-bit quantization for model weights to reduce memory",
        memory_type="semantic",
        agent_id="researcher",
        session_id="lit-review-1",
        importance=0.85,
        compute_embedding=False,
    )
    s.store_memory(
        content="Meeting notes: discussed transformer efficiency improvements",
        memory_type="episodic",
        agent_id="lab_manager",
        importance=0.5,
        compute_embedding=False,
    )

    return db_path


@pytest.fixture
def searcher(populated_store):
    return HybridSearch(db_path=populated_store, embedding_manager=None)


class TestKeywordSearch:
    def test_basic_search(self, searcher):
        results = searcher.search("transformer", mode="keyword")
        assert len(results) >= 1
        contents = [r["content"] for r in results]
        assert any("Transformers" in c or "transformer" in c for c in contents)

    def test_search_with_agent_filter(self, searcher):
        results = searcher.search("transformer", mode="keyword", agent_id="researcher")
        for r in results:
            assert r["agent_id"] == "researcher"

    def test_search_with_type_filter(self, searcher):
        results = searcher.search("memory", mode="keyword", memory_type="procedural")
        for r in results:
            assert r["memory_type"] == "procedural"

    def test_search_with_importance_filter(self, searcher):
        results = searcher.search("transformer", mode="keyword", min_importance=0.8)
        for r in results:
            assert r["importance"] >= 0.8

    def test_search_no_results(self, searcher):
        results = searcher.search("nonexistent_xyzzy_term", mode="keyword")
        assert len(results) == 0

    def test_search_limit(self, searcher):
        results = searcher.search("memory", mode="keyword", limit=1)
        assert len(results) <= 1

    def test_search_returns_scores(self, searcher):
        results = searcher.search("transformer", mode="keyword")
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_search_session_filter(self, searcher):
        results = searcher.search("BitNet", mode="keyword", session_id="lit-review-1")
        assert len(results) == 1
        assert results[0]["session_id"] == "lit-review-1"


class TestSemanticSearch:
    def test_semantic_without_embeddings_raises(self, searcher):
        with pytest.raises(RuntimeError, match="embedding manager"):
            searcher.search("test", mode="semantic")


class TestHybridSearch:
    def test_hybrid_falls_back_to_keyword(self, searcher):
        # With no embedding manager, hybrid should fall back to keyword
        results = searcher.search("transformer", mode="hybrid")
        assert len(results) >= 1

    def test_invalid_mode_raises(self, searcher):
        with pytest.raises(ValueError, match="Invalid search mode"):
            searcher.search("test", mode="invalid")
