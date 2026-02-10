"""Hybrid search combining FTS5 keyword search and semantic cosine similarity."""

import sqlite3
from typing import Any, Optional

import numpy as np

from embeddings import EmbeddingManager


class HybridSearch:
    """Search engine combining FTS5 full-text and semantic vector search."""

    def __init__(
        self,
        db_path: str,
        embedding_manager: Optional[EmbeddingManager] = None,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        self.db_path = db_path
        self.embedding_manager = embedding_manager
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Execute a search with the specified mode.

        Args:
            query: Search query text.
            mode: 'hybrid', 'keyword', or 'semantic'.
            limit: Max results to return.
            agent_id: Filter by agent (None = all).
            session_id: Filter by session (None = all).
            memory_type: Filter by type (None = all).
            min_importance: Minimum importance threshold.
            time_start: ISO8601 start time filter.
            time_end: ISO8601 end time filter.

        Returns:
            List of memory dicts with 'score' field, sorted by relevance.
        """
        if mode not in ("hybrid", "keyword", "semantic"):
            raise ValueError(f"Invalid search mode: {mode!r}")

        filters = {
            "agent_id": agent_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "min_importance": min_importance,
            "time_start": time_start,
            "time_end": time_end,
        }

        if mode == "keyword":
            return self._keyword_search(query, limit, filters)
        elif mode == "semantic":
            if self.embedding_manager is None:
                raise RuntimeError(
                    "Semantic search requires an embedding manager. "
                    "Configure MEMORY_EMBEDDING_MODEL or use mode='keyword'."
                )
            return self._semantic_search(query, limit, filters)
        else:
            # Hybrid: if no embedding manager, fall back to keyword-only
            if self.embedding_manager is None:
                return self._keyword_search(query, limit, filters)
            return self._hybrid_search(query, limit, filters)

    def _build_filter_clause(
        self, filters: dict, table_alias: str = "m"
    ) -> tuple[str, list]:
        """Build WHERE clause fragments from filters."""
        clauses = []
        params: list = []

        if filters.get("agent_id"):
            clauses.append(f"{table_alias}.agent_id = ?")
            params.append(filters["agent_id"])
        if filters.get("session_id"):
            clauses.append(f"{table_alias}.session_id = ?")
            params.append(filters["session_id"])
        if filters.get("memory_type"):
            clauses.append(f"{table_alias}.memory_type = ?")
            params.append(filters["memory_type"])
        if filters.get("min_importance", 0) > 0:
            clauses.append(f"{table_alias}.importance >= ?")
            params.append(filters["min_importance"])
        if filters.get("time_start"):
            clauses.append(f"{table_alias}.created_at >= ?")
            params.append(filters["time_start"])
        if filters.get("time_end"):
            clauses.append(f"{table_alias}.created_at <= ?")
            params.append(filters["time_end"])

        sql = (" AND " + " AND ".join(clauses)) if clauses else ""
        return sql, params

    def _keyword_search(
        self, query: str, limit: int, filters: dict
    ) -> list[dict[str, Any]]:
        """FTS5 full-text search with BM25 ranking."""
        filter_sql, filter_params = self._build_filter_clause(filters)

        # Escape FTS5 special characters in query for safety
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        sql = f"""
            SELECT m.id as memory_id, m.content, m.memory_type, m.agent_id,
                   m.session_id, m.metadata, m.created_at, m.importance,
                   m.version, rank as fts_rank
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE fts.content MATCH ?{filter_sql}
            ORDER BY fts.rank
            LIMIT ?
        """

        conn = self._connect()
        try:
            rows = conn.execute(
                sql, [safe_query] + filter_params + [limit]
            ).fetchall()
        finally:
            conn.close()

        results = []
        # Normalize BM25 scores to 0-1 (BM25 rank values are negative, lower = better)
        if rows:
            scores = [abs(r["fts_rank"]) for r in rows]
            max_score = max(scores) if scores else 1.0
            max_score = max(max_score, 1e-9)

            for row in rows:
                d = dict(row)
                d["score"] = abs(d.pop("fts_rank")) / max_score
                results.append(d)

        return results

    def _semantic_search(
        self, query: str, limit: int, filters: dict
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over stored embeddings."""
        assert self.embedding_manager is not None

        query_emb = self.embedding_manager.encode([query])[0]

        filter_sql, filter_params = self._build_filter_clause(filters, "m")
        where = "WHERE m.embedding IS NOT NULL" + filter_sql

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT m.id as memory_id, m.content, m.memory_type, m.agent_id,
                           m.session_id, m.metadata, m.created_at, m.importance,
                           m.version, m.embedding
                    FROM memories m
                    {where}""",
                filter_params,
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Deserialize embeddings and compute similarities
        memories = [dict(r) for r in rows]
        corpus = np.array([
            self.embedding_manager.deserialize_embedding(m["embedding"])
            for m in memories
        ])

        similarities = self.embedding_manager.cosine_similarity(query_emb, corpus)

        for i, mem in enumerate(memories):
            mem["score"] = float(similarities[i])
            del mem["embedding"]

        # Sort by similarity descending, take top-k
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:limit]

    def _hybrid_search(
        self, query: str, limit: int, filters: dict
    ) -> list[dict[str, Any]]:
        """Weighted fusion of keyword and semantic search results."""
        fetch_k = limit * 3

        keyword_results = self._keyword_search(query, fetch_k, filters)
        semantic_results = self._semantic_search(query, fetch_k, filters)

        # Build score maps
        kw_scores = {r["memory_id"]: r["score"] for r in keyword_results}
        sem_scores = {r["memory_id"]: r["score"] for r in semantic_results}

        # Memory lookup (prefer semantic result for full data)
        lookup: dict[str, dict] = {}
        for r in keyword_results:
            lookup[r["memory_id"]] = r
        for r in semantic_results:
            lookup[r["memory_id"]] = r

        # Fuse scores
        all_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        combined = []
        for mid in all_ids:
            mem = lookup[mid]
            mem["score"] = (
                self.keyword_weight * kw_scores.get(mid, 0.0)
                + self.semantic_weight * sem_scores.get(mid, 0.0)
            )
            combined.append(mem)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query for FTS5 MATCH.

        Wraps each word in quotes to prevent FTS5 syntax errors from
        special characters. Empty/whitespace-only queries return empty string.
        """
        words = query.strip().split()
        if not words:
            return ""
        # Quote each token to avoid FTS5 syntax issues
        return " ".join(f'"{w}"' for w in words)
