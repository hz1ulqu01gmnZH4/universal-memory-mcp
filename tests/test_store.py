"""Tests for memory_store.py — CRUD, optimistic locking, links, sessions."""

import os
import tempfile

import pytest

from memory_store import MemoryNotFoundError, MemoryStore, VersionConflictError


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path, embedding_manager=None)


class TestStoreMemory:
    def test_store_and_retrieve(self, store):
        result = store.store_memory(
            content="The brain uses ~20W of power",
            memory_type="semantic",
            agent_id="researcher",
            session_id="session-1",
            metadata={"source": "textbook"},
            importance=0.8,
            compute_embedding=False,
        )
        assert "memory_id" in result
        assert result["embedding_computed"] is False

        mem = store.get_memory(result["memory_id"])
        assert mem["content"] == "The brain uses ~20W of power"
        assert mem["memory_type"] == "semantic"
        assert mem["agent_id"] == "researcher"
        assert mem["session_id"] == "session-1"
        assert mem["metadata"] == {"source": "textbook"}
        assert mem["importance"] == 0.8
        assert mem["version"] == 1

    def test_store_minimal(self, store):
        result = store.store_memory(
            content="Hello world",
            memory_type="episodic",
            compute_embedding=False,
        )
        mem = store.get_memory(result["memory_id"])
        assert mem["agent_id"] is None
        assert mem["session_id"] is None
        assert mem["importance"] == 0.5

    def test_store_empty_content_raises(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.store_memory(content="   ", memory_type="semantic", compute_embedding=False)

    def test_store_invalid_type_raises(self, store):
        with pytest.raises(ValueError, match="Invalid memory_type"):
            store.store_memory(content="test", memory_type="invalid", compute_embedding=False)

    def test_store_invalid_importance_raises(self, store):
        with pytest.raises(ValueError, match="Importance"):
            store.store_memory(
                content="test", memory_type="semantic", importance=1.5, compute_embedding=False
            )


class TestGetMemory:
    def test_not_found_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.get_memory("nonexistent-id")


class TestUpdateMemory:
    def test_update_content(self, store):
        r = store.store_memory(content="old", memory_type="semantic", compute_embedding=False)
        mid = r["memory_id"]

        update = store.update_memory(mid, content="new", compute_embedding=False)
        assert update["updated"] is True
        assert update["new_version"] == 2

        mem = store.get_memory(mid)
        assert mem["content"] == "new"
        assert mem["version"] == 2

    def test_update_importance(self, store):
        r = store.store_memory(content="test", memory_type="semantic", compute_embedding=False)
        store.update_memory(r["memory_id"], importance=0.9, compute_embedding=False)
        mem = store.get_memory(r["memory_id"])
        assert mem["importance"] == 0.9

    def test_update_metadata(self, store):
        r = store.store_memory(content="test", memory_type="semantic", compute_embedding=False)
        store.update_memory(r["memory_id"], metadata={"key": "value"}, compute_embedding=False)
        mem = store.get_memory(r["memory_id"])
        assert mem["metadata"] == {"key": "value"}

    def test_optimistic_locking_success(self, store):
        r = store.store_memory(content="v1", memory_type="semantic", compute_embedding=False)
        mid = r["memory_id"]
        result = store.update_memory(mid, content="v2", expected_version=1, compute_embedding=False)
        assert result["new_version"] == 2

    def test_optimistic_locking_conflict(self, store):
        r = store.store_memory(content="v1", memory_type="semantic", compute_embedding=False)
        mid = r["memory_id"]
        store.update_memory(mid, content="v2", compute_embedding=False)  # bumps to version 2

        with pytest.raises(VersionConflictError):
            store.update_memory(mid, content="v3", expected_version=1, compute_embedding=False)

    def test_update_nonexistent_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.update_memory("fake-id", content="x", compute_embedding=False)

    def test_noop_update(self, store):
        r = store.store_memory(content="test", memory_type="semantic", compute_embedding=False)
        result = store.update_memory(r["memory_id"], compute_embedding=False)
        assert result["updated"] is False


class TestDeleteMemory:
    def test_delete(self, store):
        r = store.store_memory(content="to delete", memory_type="episodic", compute_embedding=False)
        result = store.delete_memory(r["memory_id"])
        assert result["status"] == "deleted"

        with pytest.raises(MemoryNotFoundError):
            store.get_memory(r["memory_id"])

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(MemoryNotFoundError):
            store.delete_memory("fake-id")


class TestLinks:
    def test_link_and_traverse(self, store):
        r1 = store.store_memory(content="cause", memory_type="episodic", compute_embedding=False)
        r2 = store.store_memory(content="effect", memory_type="episodic", compute_embedding=False)

        link = store.link_memories(r1["memory_id"], r2["memory_id"], "caused_by")
        assert link["status"] == "created"

        # Forward traversal from r1
        linked = store.get_linked_memories(r1["memory_id"], direction="forward")
        assert len(linked) == 1
        assert linked[0]["memory_id"] == r2["memory_id"]
        assert linked[0]["direction"] == "forward"

        # Backward traversal from r2
        linked = store.get_linked_memories(r2["memory_id"], direction="backward")
        assert len(linked) == 1
        assert linked[0]["memory_id"] == r1["memory_id"]

        # Both directions from r1
        linked = store.get_linked_memories(r1["memory_id"], direction="both")
        assert len(linked) == 1

    def test_link_depth_traversal(self, store):
        ids = []
        for i in range(4):
            r = store.store_memory(content=f"node-{i}", memory_type="semantic", compute_embedding=False)
            ids.append(r["memory_id"])

        # Chain: 0 -> 1 -> 2 -> 3
        for i in range(3):
            store.link_memories(ids[i], ids[i + 1], "follows")

        # Depth 1 from node 0
        linked = store.get_linked_memories(ids[0], direction="forward", max_depth=1)
        assert len(linked) == 1

        # Depth 3 from node 0 should reach all
        linked = store.get_linked_memories(ids[0], direction="forward", max_depth=3)
        assert len(linked) == 3

    def test_duplicate_link_raises(self, store):
        r1 = store.store_memory(content="a", memory_type="semantic", compute_embedding=False)
        r2 = store.store_memory(content="b", memory_type="semantic", compute_embedding=False)
        store.link_memories(r1["memory_id"], r2["memory_id"], "related_to")

        with pytest.raises(ValueError, match="already exists"):
            store.link_memories(r1["memory_id"], r2["memory_id"], "related_to")

    def test_link_cascade_delete(self, store):
        r1 = store.store_memory(content="a", memory_type="semantic", compute_embedding=False)
        r2 = store.store_memory(content="b", memory_type="semantic", compute_embedding=False)
        store.link_memories(r1["memory_id"], r2["memory_id"], "related_to")

        store.delete_memory(r1["memory_id"])
        # r2 should still exist but have no links
        linked = store.get_linked_memories(r2["memory_id"], direction="both")
        assert len(linked) == 0


class TestSessions:
    def test_create_and_list(self, store):
        result = store.create_session(agent_id="lab_manager")
        assert result["status"] == "created"
        sid = result["session_id"]

        sessions = store.list_sessions(agent_id="lab_manager")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sid

    def test_resume_session(self, store):
        r = store.create_session(agent_id="pi", session_id="my-session")
        assert r["status"] == "created"

        r2 = store.create_session(session_id="my-session")
        assert r2["status"] == "resumed"

    def test_checkpoint_and_restore(self, store):
        sess = store.create_session(agent_id="pi")
        sid = sess["session_id"]

        # Checkpoint 1
        cp1 = store.checkpoint_session(sid, state={"step": 1, "notes": "started"})
        assert cp1["checkpoint_number"] == 1

        # Checkpoint 2
        cp2 = store.checkpoint_session(sid, state={"step": 2, "notes": "progress"})
        assert cp2["checkpoint_number"] == 2

        # Restore latest (should be checkpoint 2)
        restored = store.restore_session(sid)
        assert restored["checkpoint_number"] == 2
        assert restored["state"]["step"] == 2

        # Restore specific (checkpoint 1)
        restored = store.restore_session(sid, checkpoint_id=cp1["checkpoint_id"])
        assert restored["checkpoint_number"] == 1
        assert restored["state"]["step"] == 1

    def test_restore_no_checkpoint_raises(self, store):
        sess = store.create_session(agent_id="pi")
        with pytest.raises(ValueError, match="No checkpoint"):
            store.restore_session(sess["session_id"])


class TestSchemaMigration:
    def test_migration_adds_missing_columns(self, tmp_path):
        """Simulate an old DB without new columns — migration should add them."""
        import sqlite3
        db_path = str(tmp_path / "old.db")

        # Create a minimal old-schema DB without new columns
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            session_id TEXT,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            updated_by TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            importance REAL NOT NULL DEFAULT 0.5
        )""")
        conn.commit()
        conn.close()

        # MemoryStore should migrate — not crash
        store = MemoryStore(db_path=db_path, embedding_manager=None)

        # Verify new columns exist by inserting/reading
        r = store.store_memory(content="test", memory_type="semantic", compute_embedding=False)
        mem = store.get_memory(r["memory_id"])
        assert mem["access_count"] == 1  # get_memory bumps from 0 to 1
        assert mem["last_accessed_at"] is not None


class TestContradictionDetection:
    def test_no_contradictions_without_embeddings(self, store):
        r = store.store_memory(content="fact A", memory_type="semantic", compute_embedding=False)
        assert "contradictions" not in r

    def test_contradiction_detection_with_mock_embeddings(self, tmp_path):
        """Test contradiction detection using a mock embedding manager."""
        import numpy as np
        from unittest.mock import MagicMock

        mock_emb = MagicMock()
        mock_emb.dimension = 4
        # All memories get nearly identical embeddings → should trigger contradiction
        mock_emb.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        mock_emb.serialize_embedding.side_effect = lambda e: e.astype(np.float32).tobytes()
        mock_emb.deserialize_embedding.side_effect = lambda b: np.frombuffer(b, dtype=np.float32)
        mock_emb.estimate_variance.return_value = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        mock_emb.serialize_variance.side_effect = lambda v: v.astype(np.float32).tobytes()
        mock_emb.cosine_similarity.side_effect = lambda q, c: c @ q

        db_path = str(tmp_path / "contra.db")
        store = MemoryStore(db_path=db_path, embedding_manager=mock_emb, contradiction_threshold=0.9)

        # Store first memory
        r1 = store.store_memory(content="The sky is blue", memory_type="semantic")
        assert "contradictions" not in r1  # no existing memories to contradict

        # Store second memory with same embedding → contradiction
        r2 = store.store_memory(content="The sky is green", memory_type="semantic")
        assert "contradictions" in r2
        assert len(r2["contradictions"]) == 1
        assert r2["contradictions"][0]["memory_id"] == r1["memory_id"]
        assert r2["contradictions"][0]["similarity"] >= 0.9

    def test_no_contradiction_across_types(self, tmp_path):
        """Memories of different types should not trigger contradictions."""
        import numpy as np
        from unittest.mock import MagicMock

        mock_emb = MagicMock()
        mock_emb.dimension = 4
        mock_emb.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        mock_emb.serialize_embedding.side_effect = lambda e: e.astype(np.float32).tobytes()
        mock_emb.deserialize_embedding.side_effect = lambda b: np.frombuffer(b, dtype=np.float32)
        mock_emb.estimate_variance.return_value = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        mock_emb.serialize_variance.side_effect = lambda v: v.astype(np.float32).tobytes()
        mock_emb.cosine_similarity.side_effect = lambda q, c: c @ q

        db_path = str(tmp_path / "cross.db")
        store = MemoryStore(db_path=db_path, embedding_manager=mock_emb)

        store.store_memory(content="fact", memory_type="semantic")
        r2 = store.store_memory(content="procedure", memory_type="procedural")
        assert "contradictions" not in r2


class TestStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_memories"] == 0
        assert stats["total_links"] == 0
        assert stats["total_sessions"] == 0

    def test_stats_with_data(self, store):
        store.store_memory(content="a", memory_type="episodic", agent_id="pi", compute_embedding=False)
        store.store_memory(content="b", memory_type="semantic", agent_id="pi", compute_embedding=False)
        store.store_memory(content="c", memory_type="semantic", compute_embedding=False)

        stats = store.get_stats()
        assert stats["total_memories"] == 3
        assert stats["by_type"]["episodic"] == 1
        assert stats["by_type"]["semantic"] == 2
        assert stats["by_agent"]["pi"] == 2
        assert stats["by_agent"]["_shared"] == 1
