"""Universal Memory MCP Server — persistent memory for single and multi-agent systems."""

import json
import logging
import os
from typing import Annotated, Any, Literal, Optional

from fastmcp import FastMCP
from pydantic import Field
from pydantic_settings import BaseSettings

from embeddings import EmbeddingManager, create_embedding_manager
from memory_store import MemoryNotFoundError, MemoryStore, VersionConflictError
from search import HybridSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("universal-memory")


# --- Configuration ---


class MemoryConfig(BaseSettings):
    database_path: str = "./memory.db"
    embedding_backend: str = "transformers"  # 'transformers' or 'llama-server'
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    llama_server_url: str = "http://localhost:8787"
    enable_embeddings: bool = True
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6

    model_config = {"env_prefix": "MEMORY_"}


config = MemoryConfig()

# --- Initialize components ---

embedding_manager: Optional[EmbeddingManager] = None
if config.enable_embeddings:
    embedding_manager = create_embedding_manager(
        backend=config.embedding_backend,
        model_name=config.embedding_model,
        dimension=config.embedding_dimension,
        server_url=config.llama_server_url,
    )

store = MemoryStore(db_path=config.database_path, embedding_manager=embedding_manager)
searcher = HybridSearch(
    db_path=config.database_path,
    embedding_manager=embedding_manager,
    keyword_weight=config.keyword_weight,
    semantic_weight=config.semantic_weight,
)

# --- MCP Server ---

mcp = FastMCP(
    "universal-memory",
    instructions="Persistent memory for single and multi-agent LLM systems. "
    "Supports episodic/semantic/procedural memory types, hybrid search, "
    "graph links, and session checkpoints.",
    version="0.1.0",
)


# --- Memory CRUD Tools ---


@mcp.tool()
def store_memory(
    content: Annotated[str, Field(description="Memory content text to store")],
    memory_type: Annotated[
        Literal["episodic", "semantic", "procedural"],
        Field(description="Type: 'episodic' (events/logs), 'semantic' (facts/knowledge), 'procedural' (how-to/workflows)"),
    ],
    agent_id: Annotated[
        Optional[str],
        Field(description="Agent identifier (e.g. 'principal_investigator'). None = shared memory."),
    ] = None,
    session_id: Annotated[
        Optional[str],
        Field(description="Session identifier for grouping memories. None = unscoped."),
    ] = None,
    metadata: Annotated[
        Optional[str],
        Field(description="JSON string of additional metadata (task_id, experiment_id, etc.)"),
    ] = None,
    importance: Annotated[
        float,
        Field(description="Salience score 0.0-1.0 for pruning priority. Default 0.5.", ge=0.0, le=1.0),
    ] = 0.5,
) -> dict[str, Any]:
    """Store a new memory with optional agent/session scoping and semantic embedding."""
    meta_dict = json.loads(metadata) if metadata else None
    return store.store_memory(
        content=content,
        memory_type=memory_type,
        agent_id=agent_id,
        session_id=session_id,
        metadata=meta_dict,
        importance=importance,
    )


@mcp.tool()
def recall_memories(
    query: Annotated[str, Field(description="Search query (natural language or keywords)")],
    search_mode: Annotated[
        Literal["hybrid", "keyword", "semantic"],
        Field(description="Search mode: 'hybrid' (keyword+semantic), 'keyword' (FTS5 only), 'semantic' (vector only)"),
    ] = "hybrid",
    agent_id: Annotated[
        Optional[str],
        Field(description="Filter by agent ID. None = search all agents."),
    ] = None,
    session_id: Annotated[
        Optional[str],
        Field(description="Filter by session ID. None = search all sessions."),
    ] = None,
    memory_type: Annotated[
        Optional[Literal["episodic", "semantic", "procedural"]],
        Field(description="Filter by memory type. None = search all types."),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Maximum results to return.", ge=1, le=100),
    ] = 10,
    min_importance: Annotated[
        float,
        Field(description="Minimum importance score filter.", ge=0.0, le=1.0),
    ] = 0.0,
    time_start: Annotated[
        Optional[str],
        Field(description="ISO8601 start time filter (e.g. '2025-01-01T00:00:00Z')."),
    ] = None,
    time_end: Annotated[
        Optional[str],
        Field(description="ISO8601 end time filter."),
    ] = None,
) -> list[dict[str, Any]]:
    """Search memories using hybrid keyword+semantic search with filters."""
    return searcher.search(
        query=query,
        mode=search_mode,
        limit=limit,
        agent_id=agent_id,
        session_id=session_id,
        memory_type=memory_type,
        min_importance=min_importance,
        time_start=time_start,
        time_end=time_end,
    )


@mcp.tool()
def get_memory(
    memory_id: Annotated[str, Field(description="UUID of the memory to retrieve")],
) -> dict[str, Any]:
    """Get a specific memory by its ID."""
    try:
        return store.get_memory(memory_id)
    except MemoryNotFoundError as e:
        return {"error": str(e)}


@mcp.tool()
def update_memory(
    memory_id: Annotated[str, Field(description="UUID of the memory to update")],
    content: Annotated[
        Optional[str],
        Field(description="New content. None = keep existing."),
    ] = None,
    metadata: Annotated[
        Optional[str],
        Field(description="New metadata as JSON string. None = keep existing."),
    ] = None,
    importance: Annotated[
        Optional[float],
        Field(description="New importance score. None = keep existing.", ge=0.0, le=1.0),
    ] = None,
    updated_by: Annotated[
        Optional[str],
        Field(description="Agent making this update."),
    ] = None,
    expected_version: Annotated[
        Optional[int],
        Field(description="Expected version for optimistic locking. None = skip check."),
    ] = None,
) -> dict[str, Any]:
    """Update a memory with optimistic locking for concurrent agent safety."""
    try:
        meta_dict = json.loads(metadata) if metadata else None
        return store.update_memory(
            memory_id=memory_id,
            content=content,
            metadata=meta_dict,
            importance=importance,
            updated_by=updated_by,
            expected_version=expected_version,
        )
    except MemoryNotFoundError as e:
        return {"error": str(e)}
    except VersionConflictError as e:
        return {
            "error": "version_conflict",
            "message": str(e),
            "expected": e.expected,
            "actual": e.actual,
        }


@mcp.tool()
def delete_memory(
    memory_id: Annotated[str, Field(description="UUID of the memory to delete")],
) -> dict[str, Any]:
    """Delete a memory and all its graph links."""
    try:
        return store.delete_memory(memory_id)
    except MemoryNotFoundError as e:
        return {"error": str(e)}


# --- Graph Link Tools ---


@mcp.tool()
def link_memories(
    from_memory_id: Annotated[str, Field(description="Source memory UUID")],
    to_memory_id: Annotated[str, Field(description="Target memory UUID")],
    relation_type: Annotated[
        Literal["caused_by", "related_to", "contradicts", "supports", "follows"],
        Field(description="Type of relationship between memories"),
    ],
    strength: Annotated[
        float,
        Field(description="Link strength 0.0-1.0. Default 1.0.", ge=0.0, le=1.0),
    ] = 1.0,
    created_by: Annotated[
        Optional[str],
        Field(description="Agent creating this link."),
    ] = None,
) -> dict[str, Any]:
    """Create a directed link between two memories in the knowledge graph."""
    try:
        return store.link_memories(
            from_memory_id=from_memory_id,
            to_memory_id=to_memory_id,
            relation_type=relation_type,
            strength=strength,
            created_by=created_by,
        )
    except (MemoryNotFoundError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def get_linked_memories(
    memory_id: Annotated[str, Field(description="Starting memory UUID for graph traversal")],
    relation_type: Annotated[
        Optional[Literal["caused_by", "related_to", "contradicts", "supports", "follows"]],
        Field(description="Filter by relation type. None = all relations."),
    ] = None,
    direction: Annotated[
        Literal["forward", "backward", "both"],
        Field(description="Traversal direction: 'forward' (outgoing), 'backward' (incoming), 'both'."),
    ] = "both",
    max_depth: Annotated[
        int,
        Field(description="Maximum graph traversal depth. Default 1 (immediate neighbors).", ge=1, le=5),
    ] = 1,
) -> list[dict[str, Any]] | dict[str, str]:
    """Traverse the memory graph from a starting memory."""
    try:
        return store.get_linked_memories(
            memory_id=memory_id,
            relation_type=relation_type,
            direction=direction,
            max_depth=max_depth,
        )
    except MemoryNotFoundError as e:
        return {"error": str(e)}


# --- Session & Checkpoint Tools ---


@mcp.tool()
def create_session(
    agent_id: Annotated[
        Optional[str],
        Field(description="Agent running this session."),
    ] = None,
    session_id: Annotated[
        Optional[str],
        Field(description="Specific session ID to create or resume. None = auto-generate UUID."),
    ] = None,
    parent_session_id: Annotated[
        Optional[str],
        Field(description="Parent session ID for forked sessions."),
    ] = None,
    metadata: Annotated[
        Optional[str],
        Field(description="Session metadata as JSON string."),
    ] = None,
) -> dict[str, Any]:
    """Create a new session or resume an existing one for checkpoint/restore."""
    meta_dict = json.loads(metadata) if metadata else None
    return store.create_session(
        agent_id=agent_id,
        session_id=session_id,
        parent_session_id=parent_session_id,
        metadata=meta_dict,
    )


@mcp.tool()
def checkpoint_session(
    session_id: Annotated[str, Field(description="Session UUID to checkpoint")],
    state: Annotated[
        str,
        Field(description="Agent state to save as JSON string (working memory, goals, progress, etc.)"),
    ],
) -> dict[str, Any]:
    """Save a checkpoint of agent state for later resumption."""
    state_dict = json.loads(state)
    return store.checkpoint_session(session_id=session_id, state=state_dict)


@mcp.tool()
def restore_session(
    session_id: Annotated[str, Field(description="Session UUID to restore")],
    checkpoint_id: Annotated[
        Optional[str],
        Field(description="Specific checkpoint ID. None = restore latest checkpoint."),
    ] = None,
) -> dict[str, Any]:
    """Restore agent state from a session checkpoint."""
    try:
        return store.restore_session(
            session_id=session_id, checkpoint_id=checkpoint_id
        )
    except ValueError as e:
        return {"error": str(e)}


# --- Stats Tool ---


@mcp.tool()
def get_stats() -> dict[str, Any]:
    """Get memory system statistics: counts by type/agent, links, sessions, DB size."""
    stats = store.get_stats()
    if config.enable_embeddings:
        if config.embedding_backend == "llama-server":
            stats["embedding_model"] = f"llama-server ({config.llama_server_url})"
        else:
            stats["embedding_model"] = config.embedding_model
    else:
        stats["embedding_model"] = "disabled"
    stats["database_path"] = config.database_path
    return stats


# --- Entry Point ---


def main() -> None:
    logger.info("=" * 60)
    logger.info("Universal Memory MCP Server v0.1.0")
    logger.info("Database: %s", os.path.abspath(config.database_path))
    if config.enable_embeddings and config.embedding_backend == "llama-server":
        logger.info("Embeddings: llama-server at %s (dim=%d)", config.llama_server_url, config.embedding_dimension)
    else:
        logger.info("Embeddings: %s", config.embedding_model if config.enable_embeddings else "disabled")
    logger.info("Search weights: keyword=%.1f, semantic=%.1f", config.keyword_weight, config.semantic_weight)
    logger.info("=" * 60)
    mcp.run()


if __name__ == "__main__":
    main()
