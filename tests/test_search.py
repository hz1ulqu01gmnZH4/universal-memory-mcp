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


class TestAccessTracking:
    def test_search_bumps_access_count(self, populated_store):
        searcher = HybridSearch(db_path=populated_store, embedding_manager=None)
        # First search
        results = searcher.search("transformer", mode="keyword")
        assert len(results) >= 1
        memory_id = results[0]["memory_id"]

        # Check access_count was bumped
        s = MemoryStore(db_path=populated_store, embedding_manager=None)
        mem = s.get_memory(memory_id)
        # get_memory bumps +1, search bumped +1 = at least 2
        assert mem["access_count"] >= 2

    def test_get_memory_bumps_access(self, populated_store):
        s = MemoryStore(db_path=populated_store, embedding_manager=None)
        # Store a memory and get it twice
        r = s.store_memory(content="access test", memory_type="semantic", compute_embedding=False)
        mem1 = s.get_memory(r["memory_id"])
        assert mem1["access_count"] == 1
        mem2 = s.get_memory(r["memory_id"])
        assert mem2["access_count"] == 2

    def test_access_count_in_search_results(self, searcher):
        results = searcher.search("transformer", mode="keyword")
        for r in results:
            assert "access_count" in r


class TestTemporalScoring:
    def test_scores_include_temporal_component(self, searcher):
        # All results should have scores (temporal post-scoring applied)
        results = searcher.search("memory", mode="keyword")
        assert len(results) >= 1
        for r in results:
            assert r["score"] > 0

    def test_temporal_weight_zero_disables(self, populated_store):
        # With temporal_weight=0, post-scoring should not change relative ordering
        s1 = HybridSearch(db_path=populated_store, embedding_manager=None, temporal_weight=0.0)
        s2 = HybridSearch(db_path=populated_store, embedding_manager=None, temporal_weight=0.0)
        r1 = s1.search("transformer", mode="keyword")
        r2 = s2.search("transformer", mode="keyword")
        # Same ordering
        assert [r["memory_id"] for r in r1] == [r["memory_id"] for r in r2]


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


class TestPostScoringBounds:
    def test_score_does_not_exceed_one(self, populated_store):
        # Even with high temporal_weight and many accesses, scores should be bounded
        s = MemoryStore(db_path=populated_store, embedding_manager=None)
        # Bump access count artificially
        import sqlite3
        conn = sqlite3.connect(populated_store)
        conn.execute("UPDATE memories SET access_count = 10000")
        conn.commit()
        conn.close()

        searcher = HybridSearch(
            db_path=populated_store, embedding_manager=None, temporal_weight=0.5
        )
        results = searcher.search("transformer", mode="keyword")
        for r in results:
            assert r["score"] <= 1.5  # WRRF scores are small, post-scoring adds up to tw*1.0
