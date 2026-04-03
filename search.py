"""Hybrid search combining FTS5 keyword search and semantic cosine similarity."""

import math
import sqlite3
from datetime import datetime, timezone
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
        temporal_decay_lambda: float = 0.01,
        temporal_weight: float = 0.1,
    ):
        self.db_path = db_path
        self.embedding_manager = embedding_manager
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.temporal_decay_lambda = temporal_decay_lambda
        self.temporal_weight = temporal_weight

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _bump_access(self, memory_ids: list[str]) -> None:
        """Increment access_count for memories returned by search."""
        if not memory_ids:
            return
        conn = self._connect()
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            placeholders = ",".join("?" for _ in memory_ids)
            conn.execute(
                f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                f"WHERE id IN ({placeholders})",
                [now] + memory_ids,
            )
            conn.commit()
        finally:
            conn.close()

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
            results = self._keyword_search(query, limit, filters)
        elif mode == "semantic":
            if self.embedding_manager is None:
                raise RuntimeError(
                    "Semantic search requires an embedding manager. "
                    "Configure MEMORY_EMBEDDING_MODEL or use mode='keyword'."
                )
            results = self._semantic_search(query, limit, filters)
        else:
            # Hybrid: if no embedding manager, fall back to keyword-only
            if self.embedding_manager is None:
                results = self._keyword_search(query, limit, filters)
            else:
                results = self._hybrid_search(query, limit, filters)

        # Apply temporal decay and access boost to final scores
        self._apply_post_scoring(results)

        # Re-sort after post-scoring
        results.sort(key=lambda x: x["score"], reverse=True)

        # Track access for returned results
        self._bump_access([r["memory_id"] for r in results])
        return results

    def _apply_post_scoring(self, results: list[dict[str, Any]]) -> None:
        """Apply temporal decay and access-frequency boost to search results.

        Temporal decay: exp(-lambda * days_since_created) — recent memories score higher.
        Access boost: log(1 + access_count) — frequently accessed memories score higher.
        Both are blended into the existing relevance score.
        """
        if not results:
            return
        now = datetime.now(timezone.utc)
        for r in results:
            # Temporal decay
            try:
                created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                days_old = max((now - created).total_seconds() / 86400, 0)
            except (ValueError, KeyError):
                days_old = 0
            temporal_score = math.exp(-self.temporal_decay_lambda * days_old)

            # Access frequency boost (logarithmic to avoid runaway)
            access_count = r.get("access_count", 0) or 0
            access_boost = math.log1p(access_count) / 10  # normalized: log(11)/10 ≈ 0.24 at 10 accesses

            # Blend: (1 - tw) * relevance + tw * min(temporal + access_boost, 1.0)
            tw = self.temporal_weight
            r["score"] = (1 - tw) * r["score"] + tw * min(temporal_score + access_boost, 1.0)

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
                   m.version, m.access_count, rank as fts_rank
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
        """Semantic search with graduated Fisher-Rao / cosine similarity.

        Uses a graduated ramp: memories with few accesses use cosine similarity,
        memories with many accesses (stable variance estimates) use Fisher-Rao
        information-weighted similarity. Blend factor: min(access_count/10, 1).
        """
        assert self.embedding_manager is not None

        query_emb = self.embedding_manager.encode([query])[0]

        filter_sql, filter_params = self._build_filter_clause(filters, "m")
        where = "WHERE m.embedding IS NOT NULL" + filter_sql

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT m.id as memory_id, m.content, m.memory_type, m.agent_id,
                           m.session_id, m.metadata, m.created_at, m.importance,
                           m.version, m.access_count, m.embedding, m.embedding_variance
                    FROM memories m
                    {where}""",
                filter_params,
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        memories = [dict(r) for r in rows]
        corpus = np.array([
            self.embedding_manager.deserialize_embedding(m["embedding"])
            for m in memories
        ])

        # Cosine similarities (always computed)
        cos_sims = self.embedding_manager.cosine_similarity(query_emb, corpus)

        # Fisher-Rao similarities (for memories with variance data)
        has_variance = [m["embedding_variance"] is not None for m in memories]
        if any(has_variance):
            variances = np.array([
                self.embedding_manager.deserialize_variance(m["embedding_variance"])
                if m["embedding_variance"] is not None
                else self.embedding_manager.estimate_variance(corpus[i])
                for i, m in enumerate(memories)
            ])
            fr_sims = self.embedding_manager.fisher_rao_similarity(
                query_emb, corpus, variances
            )
        else:
            fr_sims = cos_sims

        # Graduated ramp: blend cosine → Fisher-Rao based on access count.
        # Note: this creates an intentional feedback loop — frequently accessed
        # memories get Fisher-Rao scoring (more discriminative), which may change
        # their ranking, affecting future access patterns. This is by design:
        # well-exercised memories earn more precise similarity scoring.
        for i, mem in enumerate(memories):
            access_count = mem.get("access_count", 0) or 0
            alpha = min(access_count / 10.0, 1.0)
            mem["score"] = float((1 - alpha) * cos_sims[i] + alpha * fr_sims[i])
            del mem["embedding"]
            if "embedding_variance" in mem:
                del mem["embedding_variance"]

        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:limit]

    def _hybrid_search(
        self, query: str, limit: int, filters: dict
    ) -> list[dict[str, Any]]:
        """Weighted Reciprocal Rank Fusion (WRRF) of keyword and semantic results.

        WRRF is more robust than linear score combination because it operates on
        ranks rather than raw scores, avoiding scale mismatch between channels.
        Formula: WRRF(m) = sum(w_i / (k + rank_i(m)))  where k=60.
        """
        fetch_k = limit * 3
        rrf_k = 60  # smoothing constant (standard value from Cormack et al.)

        keyword_results = self._keyword_search(query, fetch_k, filters)
        semantic_results = self._semantic_search(query, fetch_k, filters)

        # Build rank maps (1-indexed: rank 1 = best)
        kw_ranks = {r["memory_id"]: i + 1 for i, r in enumerate(keyword_results)}
        sem_ranks = {r["memory_id"]: i + 1 for i, r in enumerate(semantic_results)}

        # Memory lookup (prefer semantic result for full data)
        lookup: dict[str, dict] = {}
        for r in keyword_results:
            lookup[r["memory_id"]] = r
        for r in semantic_results:
            lookup[r["memory_id"]] = r

        # WRRF fusion
        all_ids = set(kw_ranks.keys()) | set(sem_ranks.keys())
        combined = []
        for mid in all_ids:
            mem = lookup[mid]
            rrf_score = 0.0
            if mid in kw_ranks:
                rrf_score += self.keyword_weight / (rrf_k + kw_ranks[mid])
            if mid in sem_ranks:
                rrf_score += self.semantic_weight / (rrf_k + sem_ranks[mid])
            mem["score"] = rrf_score
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
        # Strip double quotes from tokens, then wrap in quotes to avoid FTS5 syntax issues
        return " ".join(f'"{w.replace(chr(34), "")}"' for w in words)
