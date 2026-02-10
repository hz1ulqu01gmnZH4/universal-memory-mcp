"""Tests for server.py — verify MCP server tools are registered and work end-to-end.

FastMCP wraps tool functions into FunctionTool objects, so we call the
underlying .fn attribute directly for unit testing. The store/search
layer tests (test_store.py, test_search.py) cover the core logic.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Reinitialize server globals with a temporary database."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("MEMORY_DATABASE_PATH", db_path)
    monkeypatch.setenv("MEMORY_ENABLE_EMBEDDINGS", "false")

    if "server" in sys.modules:
        import server as srv
        from memory_store import MemoryStore
        from search import HybridSearch

        srv.config = srv.MemoryConfig()
        srv.store = MemoryStore(db_path=db_path, embedding_manager=None)
        srv.searcher = HybridSearch(db_path=db_path, embedding_manager=None)
    else:
        import server  # noqa: F401

    return db_path


def _call_tool(tool_obj, **kwargs):
    """Call a FastMCP FunctionTool's underlying function."""
    return tool_obj.fn(**kwargs)


class TestToolRegistration:
    def test_all_tools_registered(self):
        import server as srv

        tool_names = {t.name for t in srv.mcp._tool_manager._tools.values()}
        expected = {
            "store_memory", "recall_memories", "get_memory", "update_memory",
            "delete_memory", "link_memories", "get_linked_memories",
            "create_session", "checkpoint_session", "restore_session",
            "get_stats",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"


class TestStoreAndRecall:
    def test_store_and_get(self):
        import server as srv

        result = _call_tool(
            srv.store_memory,
            content="Test fact about AI",
            memory_type="semantic",
            agent_id="test-agent",
        )
        assert "memory_id" in result

        mem = _call_tool(srv.get_memory, memory_id=result["memory_id"])
        assert mem["content"] == "Test fact about AI"
        assert mem["agent_id"] == "test-agent"

    def test_store_and_recall_keyword(self):
        import server as srv

        _call_tool(srv.store_memory, content="Quantum computing uses qubits", memory_type="semantic")
        _call_tool(srv.store_memory, content="Classical computing uses bits", memory_type="semantic")

        results = _call_tool(srv.recall_memories, query="quantum", search_mode="keyword")
        assert len(results) >= 1
        assert any("Quantum" in r["content"] or "quantum" in r["content"] for r in results)

    def test_store_with_metadata(self):
        import server as srv

        result = _call_tool(
            srv.store_memory,
            content="Experiment result",
            memory_type="episodic",
            metadata='{"experiment_id": "exp-42", "accuracy": 0.95}',
        )
        mem = _call_tool(srv.get_memory, memory_id=result["memory_id"])
        assert mem["metadata"]["experiment_id"] == "exp-42"


class TestUpdateAndDelete:
    def test_update(self):
        import server as srv

        r = _call_tool(srv.store_memory, content="old content", memory_type="semantic")
        _call_tool(srv.update_memory, memory_id=r["memory_id"], content="new content")
        mem = _call_tool(srv.get_memory, memory_id=r["memory_id"])
        assert mem["content"] == "new content"

    def test_delete(self):
        import server as srv

        r = _call_tool(srv.store_memory, content="to delete", memory_type="episodic")
        _call_tool(srv.delete_memory, memory_id=r["memory_id"])
        result = _call_tool(srv.get_memory, memory_id=r["memory_id"])
        assert "error" in result

    def test_version_conflict(self):
        import server as srv

        r = _call_tool(srv.store_memory, content="v1", memory_type="semantic")
        _call_tool(srv.update_memory, memory_id=r["memory_id"], content="v2")
        result = _call_tool(
            srv.update_memory, memory_id=r["memory_id"], content="v3", expected_version=1
        )
        assert result["error"] == "version_conflict"


class TestLinks:
    def test_link_and_traverse(self):
        import server as srv

        r1 = _call_tool(srv.store_memory, content="hypothesis A", memory_type="semantic")
        r2 = _call_tool(srv.store_memory, content="evidence for A", memory_type="episodic")

        link = _call_tool(
            srv.link_memories,
            from_memory_id=r1["memory_id"],
            to_memory_id=r2["memory_id"],
            relation_type="supports",
        )
        assert link["status"] == "created"

        linked = _call_tool(srv.get_linked_memories, memory_id=r1["memory_id"])
        assert len(linked) == 1
        assert linked[0]["memory_id"] == r2["memory_id"]


class TestSessions:
    def test_session_lifecycle(self):
        import server as srv

        sess = _call_tool(srv.create_session, agent_id="pi")
        sid = sess["session_id"]

        _call_tool(
            srv.checkpoint_session,
            session_id=sid,
            state='{"current_task": "literature review", "progress": 0.5}',
        )

        restored = _call_tool(srv.restore_session, session_id=sid)
        assert restored["state"]["current_task"] == "literature review"

    def test_restore_no_checkpoint(self):
        import server as srv

        sess = _call_tool(srv.create_session, agent_id="pi")
        result = _call_tool(srv.restore_session, session_id=sess["session_id"])
        assert "error" in result


class TestStats:
    def test_stats(self):
        import server as srv

        _call_tool(srv.store_memory, content="a", memory_type="episodic")
        _call_tool(srv.store_memory, content="b", memory_type="semantic")

        stats = _call_tool(srv.get_stats)
        assert stats["total_memories"] == 2
        assert stats["embedding_model"] == "disabled"
