"""Core memory storage operations with SQLite backend and optimistic locking."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class VersionConflictError(Exception):
    """Raised when optimistic locking detects a version mismatch."""

    def __init__(self, memory_id: str, expected: int, actual: int):
        self.memory_id = memory_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Version conflict on memory {memory_id}: "
            f"expected {expected}, found {actual}"
        )


class MemoryNotFoundError(Exception):
    """Raised when a memory ID does not exist."""

    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class MemoryStore:
    """SQLite-backed memory storage with FTS5 and optimistic locking."""

    def __init__(self, db_path: str, embedding_manager: Optional[Any] = None):
        self.db_path = db_path
        self.embedding_manager = embedding_manager
        self._ensure_db_dir()
        self._init_schema()

    def _ensure_db_dir(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text()
        conn = self._connect()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def store_memory(
        self,
        content: str,
        memory_type: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        compute_embedding: bool = True,
    ) -> dict:
        """Store a new memory. Returns dict with memory_id, created_at, embedding_computed."""
        if memory_type not in ("episodic", "semantic", "procedural"):
            raise ValueError(f"Invalid memory_type: {memory_type!r}")
        if not content.strip():
            raise ValueError("Content must not be empty")
        if not (0.0 <= importance <= 1.0):
            raise ValueError(f"Importance must be 0.0-1.0, got {importance}")

        memory_id = str(uuid.uuid4())
        now = self._now_iso()
        metadata_json = json.dumps(metadata) if metadata else None

        embedding_blob = None
        variance_blob = None
        embedding_computed = False
        if compute_embedding and self.embedding_manager is not None:
            try:
                emb = self.embedding_manager.encode([content])[0]
                embedding_blob = self.embedding_manager.serialize_embedding(emb)
                variance = self.embedding_manager.estimate_variance(emb)
                variance_blob = self.embedding_manager.serialize_variance(variance)
                embedding_computed = True
            except Exception as e:
                raise RuntimeError(f"Embedding computation failed: {e}") from e

        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO memories
                   (id, agent_id, session_id, memory_type, content, embedding,
                    embedding_variance, metadata, created_at, updated_at, version, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (
                    memory_id, agent_id, session_id, memory_type, content,
                    embedding_blob, variance_blob, metadata_json, now, now, importance,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Detect contradictions with existing memories
        contradictions = []
        if embedding_computed and self.embedding_manager is not None:
            emb = self.embedding_manager.deserialize_embedding(embedding_blob)
            contradictions = self._detect_contradictions(
                memory_id=memory_id,
                embedding=emb,
                memory_type=memory_type,
                agent_id=agent_id,
            )

        result = {
            "memory_id": memory_id,
            "created_at": now,
            "embedding_computed": embedding_computed,
        }
        if contradictions:
            result["contradictions"] = contradictions
        return result

    def get_memory(self, memory_id: str) -> dict:
        """Get a single memory by ID. Raises MemoryNotFoundError if missing."""
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id, agent_id, session_id, memory_type, content,
                          metadata, created_at, updated_at, updated_by,
                          version, importance, access_count, last_accessed_at
                   FROM memories WHERE id = ?""",
                (memory_id,),
            ).fetchone()

            if row is None:
                raise MemoryNotFoundError(memory_id)

            # Bump access tracking
            now = self._now_iso()
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                (now, memory_id),
            )
            conn.commit()
        finally:
            conn.close()

        result = dict(row)
        result["access_count"] = result["access_count"] + 1
        result["last_accessed_at"] = now
        if result["metadata"]:
            result["metadata"] = json.loads(result["metadata"])
        return result

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
        importance: Optional[float] = None,
        updated_by: Optional[str] = None,
        expected_version: Optional[int] = None,
        compute_embedding: bool = True,
    ) -> dict:
        """Update a memory with optimistic locking. Returns new version info."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version, content FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()

            if row is None:
                raise MemoryNotFoundError(memory_id)

            current_version = row["version"]
            if expected_version is not None and current_version != expected_version:
                raise VersionConflictError(memory_id, expected_version, current_version)

            updates = []
            params = []
            now = self._now_iso()

            if content is not None:
                if not content.strip():
                    raise ValueError("Content must not be empty")
                updates.append("content = ?")
                params.append(content)

                if compute_embedding and self.embedding_manager is not None:
                    try:
                        emb = self.embedding_manager.encode([content])[0]
                        embedding_blob = self.embedding_manager.serialize_embedding(emb)
                        updates.append("embedding = ?")
                        params.append(embedding_blob)
                        variance = self.embedding_manager.estimate_variance(emb)
                        variance_blob = self.embedding_manager.serialize_variance(variance)
                        updates.append("embedding_variance = ?")
                        params.append(variance_blob)
                    except Exception as e:
                        raise RuntimeError(f"Embedding computation failed: {e}") from e

            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            if importance is not None:
                if not (0.0 <= importance <= 1.0):
                    raise ValueError(f"Importance must be 0.0-1.0, got {importance}")
                updates.append("importance = ?")
                params.append(importance)

            if updated_by is not None:
                updates.append("updated_by = ?")
                params.append(updated_by)

            if not updates:
                return {
                    "memory_id": memory_id,
                    "new_version": current_version,
                    "updated": False,
                }

            new_version = current_version + 1
            updates.append("version = ?")
            params.append(new_version)
            updates.append("updated_at = ?")
            params.append(now)

            params.append(memory_id)
            params.append(current_version)

            result = conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ? AND version = ?",
                params,
            )

            if result.rowcount == 0:
                # Another writer changed it between our SELECT and UPDATE
                raise VersionConflictError(memory_id, current_version, -1)

            conn.commit()
        finally:
            conn.close()

        return {
            "memory_id": memory_id,
            "new_version": new_version,
            "updated": True,
            "updated_at": now,
        }

    def delete_memory(self, memory_id: str) -> dict:
        """Delete a memory and its links (CASCADE). Raises MemoryNotFoundError if missing."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory_id)

            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        finally:
            conn.close()

        return {"memory_id": memory_id, "status": "deleted"}

    # --- Contradiction detection ---

    def _detect_contradictions(
        self,
        memory_id: str,
        embedding: np.ndarray,
        memory_type: str,
        agent_id: Optional[str],
        similarity_threshold: float = 0.85,
        max_candidates: int = 50,
    ) -> list[dict]:
        """Detect potential contradictions with existing memories.

        Finds memories with high semantic similarity to the new memory.
        High similarity + same type suggests potential contradiction (updated fact,
        revised procedure, etc.). Auto-creates 'contradicts' links.

        Returns list of contradiction info dicts.
        """
        if self.embedding_manager is None:
            return []

        conn = self._connect()
        try:
            # Fetch candidate memories with embeddings (same type, optionally same agent)
            params: list = []
            agent_filter = ""
            if agent_id is not None:
                agent_filter = " AND agent_id = ?"
                params.append(agent_id)

            rows = conn.execute(
                f"""SELECT id, content, embedding, memory_type
                    FROM memories
                    WHERE embedding IS NOT NULL AND id != ?
                      AND memory_type = ?{agent_filter}
                    ORDER BY created_at DESC LIMIT ?""",
                [memory_id, memory_type] + params + [max_candidates],
            ).fetchall()

            if not rows:
                return []

            # Compute similarities
            corpus = np.array([
                self.embedding_manager.deserialize_embedding(r["embedding"])
                for r in rows
            ])
            similarities = self.embedding_manager.cosine_similarity(embedding, corpus)

            # Find high-similarity candidates
            contradictions = []
            for i, row in enumerate(rows):
                sim = float(similarities[i])
                if sim >= similarity_threshold:
                    # Auto-create contradicts link (new supersedes old)
                    try:
                        self.link_memories(
                            from_memory_id=memory_id,
                            to_memory_id=row["id"],
                            relation_type="contradicts",
                            strength=sim,
                        )
                        contradictions.append({
                            "memory_id": row["id"],
                            "similarity": round(sim, 4),
                            "content_preview": row["content"][:100],
                        })
                    except (ValueError, MemoryNotFoundError):
                        pass  # duplicate link or deleted memory — skip
        finally:
            conn.close()

        if contradictions:
            logger.info(
                "Detected %d potential contradiction(s) for memory %s",
                len(contradictions), memory_id,
            )
        return contradictions

    # --- Link operations ---

    def link_memories(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relation_type: str,
        strength: float = 1.0,
        created_by: Optional[str] = None,
    ) -> dict:
        """Create a link between two memories."""
        valid_relations = ("caused_by", "related_to", "contradicts", "supports", "follows")
        if relation_type not in valid_relations:
            raise ValueError(f"Invalid relation_type: {relation_type!r}")
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Strength must be 0.0-1.0, got {strength}")

        link_id = str(uuid.uuid4())
        now = self._now_iso()

        conn = self._connect()
        try:
            # Verify both memories exist
            for mid in (from_memory_id, to_memory_id):
                if conn.execute("SELECT 1 FROM memories WHERE id = ?", (mid,)).fetchone() is None:
                    raise MemoryNotFoundError(mid)

            conn.execute(
                """INSERT INTO memory_links
                   (id, from_memory_id, to_memory_id, relation_type, strength, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (link_id, from_memory_id, to_memory_id, relation_type, strength, now, created_by),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint" in str(e):
                raise ValueError(
                    f"Link already exists: {from_memory_id} -[{relation_type}]-> {to_memory_id}"
                ) from e
            raise
        finally:
            conn.close()

        return {"link_id": link_id, "status": "created"}

    def get_linked_memories(
        self,
        memory_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[dict]:
        """Traverse memory graph via BFS up to max_depth."""
        if direction not in ("forward", "backward", "both"):
            raise ValueError(f"Invalid direction: {direction!r}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")

        conn = self._connect()
        try:
            # Verify starting memory exists
            if conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone() is None:
                raise MemoryNotFoundError(memory_id)

            visited = {memory_id}
            results = []
            current_frontier = [memory_id]

            for depth in range(1, max_depth + 1):
                next_frontier = []
                for node_id in current_frontier:
                    neighbors = self._get_neighbors(
                        conn, node_id, relation_type, direction
                    )
                    for neighbor in neighbors:
                        nid = neighbor["memory_id"]
                        if nid not in visited:
                            visited.add(nid)
                            neighbor["depth"] = depth
                            results.append(neighbor)
                            next_frontier.append(nid)
                current_frontier = next_frontier
                if not current_frontier:
                    break
        finally:
            conn.close()

        return results

    def _get_neighbors(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        relation_type: Optional[str],
        direction: str,
    ) -> list[dict]:
        """Get direct neighbors of a memory node."""
        results = []
        relation_filter = ""
        params_extra: list = []
        if relation_type:
            relation_filter = " AND ml.relation_type = ?"
            params_extra = [relation_type]

        if direction in ("forward", "both"):
            rows = conn.execute(
                f"""SELECT m.id, m.content, m.memory_type, m.agent_id,
                           ml.relation_type, ml.strength
                    FROM memory_links ml
                    JOIN memories m ON m.id = ml.to_memory_id
                    WHERE ml.from_memory_id = ?{relation_filter}""",
                [memory_id] + params_extra,
            ).fetchall()
            for r in rows:
                results.append({**dict(r), "memory_id": r["id"], "direction": "forward"})

        if direction in ("backward", "both"):
            rows = conn.execute(
                f"""SELECT m.id, m.content, m.memory_type, m.agent_id,
                           ml.relation_type, ml.strength
                    FROM memory_links ml
                    JOIN memories m ON m.id = ml.from_memory_id
                    WHERE ml.to_memory_id = ?{relation_filter}""",
                [memory_id] + params_extra,
            ).fetchall()
            for r in rows:
                results.append({**dict(r), "memory_id": r["id"], "direction": "backward"})

        return results

    # --- Session operations ---

    def create_session(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create or resume a session."""
        sid = session_id or str(uuid.uuid4())
        now = self._now_iso()

        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (sid,)
            ).fetchone()

            if existing:
                # Resume: update timestamp
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid)
                )
                conn.commit()
                return {"session_id": sid, "status": "resumed", "updated_at": now}

            conn.execute(
                """INSERT INTO sessions (id, agent_id, parent_session_id, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, agent_id, parent_session_id, now, now,
                 json.dumps(metadata) if metadata else None),
            )
            conn.commit()
        finally:
            conn.close()

        return {"session_id": sid, "status": "created", "created_at": now}

    def checkpoint_session(
        self,
        session_id: str,
        state: dict,
        memory_snapshot: Optional[list[str]] = None,
    ) -> dict:
        """Save a checkpoint for a session."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Session not found: {session_id}")

            # Get next checkpoint number
            max_row = conn.execute(
                "SELECT COALESCE(MAX(checkpoint_number), 0) as max_num FROM checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_num = max_row["max_num"] + 1

            checkpoint_id = str(uuid.uuid4())
            now = self._now_iso()

            conn.execute(
                """INSERT INTO checkpoints (id, session_id, checkpoint_number, state, memory_snapshot, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id, session_id, next_num,
                    json.dumps(state),
                    json.dumps(memory_snapshot) if memory_snapshot else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ?, state = ? WHERE id = ?",
                (now, json.dumps(state), session_id),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "checkpoint_id": checkpoint_id,
            "checkpoint_number": next_num,
            "session_id": session_id,
            "created_at": now,
        }

    def restore_session(
        self, session_id: str, checkpoint_id: Optional[str] = None
    ) -> dict:
        """Restore a session from a checkpoint (default: latest)."""
        conn = self._connect()
        try:
            if checkpoint_id:
                row = conn.execute(
                    """SELECT id, checkpoint_number, state, memory_snapshot, created_at
                       FROM checkpoints WHERE id = ? AND session_id = ?""",
                    (checkpoint_id, session_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT id, checkpoint_number, state, memory_snapshot, created_at
                       FROM checkpoints WHERE session_id = ?
                       ORDER BY checkpoint_number DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise ValueError(
                f"No checkpoint found for session {session_id}"
                + (f" with id {checkpoint_id}" if checkpoint_id else "")
            )

        result = dict(row)
        result["state"] = json.loads(result["state"])
        if result["memory_snapshot"]:
            result["memory_snapshot"] = json.loads(result["memory_snapshot"])
        return result

    def list_sessions(
        self, agent_id: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """List sessions, optionally filtered by agent."""
        conn = self._connect()
        try:
            if agent_id:
                rows = conn.execute(
                    """SELECT s.id as session_id, s.agent_id, s.created_at, s.updated_at, s.metadata,
                              (SELECT COUNT(*) FROM checkpoints c WHERE c.session_id = s.id) as checkpoint_count
                       FROM sessions s WHERE s.agent_id = ?
                       ORDER BY s.updated_at DESC LIMIT ?""",
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT s.id as session_id, s.agent_id, s.created_at, s.updated_at, s.metadata,
                              (SELECT COUNT(*) FROM checkpoints c WHERE c.session_id = s.id) as checkpoint_count
                       FROM sessions s
                       ORDER BY s.updated_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            d = dict(row)
            if d["metadata"]:
                d["metadata"] = json.loads(d["metadata"])
            results.append(d)
        return results

    def get_stats(self) -> dict:
        """Get memory system statistics."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]

            by_type = {}
            for row in conn.execute(
                "SELECT memory_type, COUNT(*) as c FROM memories GROUP BY memory_type"
            ).fetchall():
                by_type[row["memory_type"]] = row["c"]

            by_agent = {}
            for row in conn.execute(
                "SELECT COALESCE(agent_id, '_shared') as aid, COUNT(*) as c FROM memories GROUP BY agent_id"
            ).fetchall():
                by_agent[row["aid"]] = row["c"]

            total_links = conn.execute("SELECT COUNT(*) as c FROM memory_links").fetchone()["c"]
            total_sessions = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
            total_checkpoints = conn.execute("SELECT COUNT(*) as c FROM checkpoints").fetchone()["c"]

            # DB file size
            db_size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        finally:
            conn.close()

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_agent": by_agent,
            "total_links": total_links,
            "total_sessions": total_sessions,
            "total_checkpoints": total_checkpoints,
            "database_size_mb": round(db_size_bytes / (1024 * 1024), 2),
        }
