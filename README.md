# universal-memory-mcp

Persistent memory MCP server for single and multi-agent LLM systems. Gives AI agents long-term memory backed by SQLite with hybrid keyword + semantic search.

## Features

- **Memory types**: episodic (events/logs), semantic (facts/knowledge), procedural (how-to/workflows)
- **Hybrid search**: FTS5 keyword search + cosine similarity over embeddings, with configurable weights
- **Knowledge graph**: directed links between memories (caused_by, related_to, contradicts, supports, follows) with BFS traversal
- **Session checkpoints**: save/restore agent state across conversations
- **Multi-agent support**: scope memories by agent_id, session_id, or share globally
- **Optimistic locking**: safe concurrent updates with version conflict detection
- **Pluggable embeddings**: HuggingFace transformers (in-process) or llama-server (external HTTP)

## Install

```bash
uv sync
```

## Usage

Run as an MCP server (stdio transport):

```bash
uv run python server.py
```

Or via the wrapper script:

```bash
./run.sh
```

### Claude Code config

Add to your MCP settings (`~/.claude/settings.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/universal-memory-mcp", "python", "server.py"]
    }
  }
}
```

## Configuration

All settings via environment variables (prefix `MEMORY_`):

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DATABASE_PATH` | `./memory.db` | SQLite database path |
| `MEMORY_EMBEDDING_BACKEND` | `transformers` | `transformers` or `llama-server` |
| `MEMORY_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model name |
| `MEMORY_EMBEDDING_DIMENSION` | `384` | Embedding vector size |
| `MEMORY_LLAMA_SERVER_URL` | `http://localhost:8787` | llama-server endpoint |
| `MEMORY_ENABLE_EMBEDDINGS` | `true` | Set `false` for keyword-only search |
| `MEMORY_KEYWORD_WEIGHT` | `0.4` | Hybrid search keyword weight |
| `MEMORY_SEMANTIC_WEIGHT` | `0.6` | Hybrid search semantic weight |

### Using llama-server backend

For lower memory usage with a GGUF model:

```bash
llama-server --model embeddinggemma-300m-Q4_0.gguf --port 8787 --embedding --ctx-size 512
MEMORY_EMBEDDING_BACKEND=llama-server MEMORY_EMBEDDING_DIMENSION=768 uv run python server.py
```

## MCP Tools

| Tool | Description |
|---|---|
| `store_memory` | Store a memory with type, agent/session scope, importance |
| `recall_memories` | Hybrid/keyword/semantic search with filters |
| `get_memory` | Retrieve a memory by ID |
| `update_memory` | Update with optimistic locking |
| `delete_memory` | Delete a memory and its links |
| `link_memories` | Create directed graph links between memories |
| `get_linked_memories` | Traverse the memory graph (BFS) |
| `create_session` | Create or resume a session |
| `checkpoint_session` | Save agent state checkpoint |
| `restore_session` | Restore from checkpoint |
| `get_stats` | Memory system statistics |

## Tests

```bash
uv run pytest
```

## License

[WTFPL](LICENSE)
