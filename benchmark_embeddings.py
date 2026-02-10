"""Benchmark: MiniLM-L6-v2 (transformers) vs EmbeddingGemma-300M (GGUF/llama.cpp)

Compares:
- Load time
- Embedding dimensions
- Single-text latency
- Batch latency (10, 100 texts)
- Memory usage (RSS)
- Embedding quality (cosine similarity on known-similar pairs)
"""

import gc
import os
import resource
import time
from pathlib import Path

import numpy as np

# Test corpus
SINGLE_TEXT = "Transformers use self-attention mechanisms for sequence processing"

BATCH_10 = [
    "Transformers use O(n^2) attention complexity for sequence processing",
    "The brain operates at approximately 20 watts of power consumption",
    "BitNet uses 1-bit quantization for model weights to reduce memory usage",
    "Reinforcement learning from human feedback aligns language models",
    "Gradient checkpointing trades compute for memory in backpropagation",
    "Knowledge distillation transfers learning from large to small models",
    "Flash attention reduces memory from O(n^2) to O(n) for attention",
    "Mixture of experts routes tokens to specialized sub-networks",
    "Chain of thought prompting improves reasoning in large language models",
    "Low-rank adaptation fine-tunes models by learning small weight deltas",
]

BATCH_100 = BATCH_10 * 10  # Repeat to get 100 texts

# Similarity test pairs (should be high similarity)
SIMILAR_PAIRS = [
    ("Neural networks learn hierarchical representations of data",
     "Deep learning models build layered feature representations"),
    ("The cat sat on the mat",
     "A feline rested on the rug"),
    ("CUDA out of memory error during training",
     "GPU ran out of VRAM while training the model"),
]

# Dissimilar pairs (should be low similarity)
DISSIMILAR_PAIRS = [
    ("Neural networks learn hierarchical representations",
     "The weather forecast predicts rain tomorrow"),
    ("Quantum entanglement enables non-local correlations",
     "The recipe calls for two cups of flour"),
]


def get_rss_mb():
    """Get current RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def benchmark_minilm():
    """Benchmark sentence-transformers/all-MiniLM-L6-v2 via transformers."""
    print("\n" + "=" * 70)
    print("MODEL: sentence-transformers/all-MiniLM-L6-v2 (transformers)")
    print("=" * 70)

    from embeddings import EmbeddingManager

    rss_before = get_rss_mb()

    # Load time
    t0 = time.perf_counter()
    mgr = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    )
    # Force model load
    mgr.encode(["warmup"])
    load_time = time.perf_counter() - t0
    rss_after = get_rss_mb()

    print(f"Load time:        {load_time:.3f}s")
    print(f"Memory (RSS):     {rss_after - rss_before:.1f} MB (delta), {rss_after:.1f} MB (total)")

    # Dimensions
    emb = mgr.encode([SINGLE_TEXT])[0]
    print(f"Dimensions:       {len(emb)}")

    # Single text latency (average of 10 runs)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        mgr.encode([SINGLE_TEXT])
        times.append(time.perf_counter() - t0)
    print(f"Single latency:   {np.mean(times)*1000:.1f}ms (mean), {np.std(times)*1000:.1f}ms (std)")

    # Batch 10
    t0 = time.perf_counter()
    mgr.encode(BATCH_10)
    t_batch10 = time.perf_counter() - t0
    print(f"Batch-10 latency: {t_batch10*1000:.1f}ms ({t_batch10/10*1000:.1f}ms/text)")

    # Batch 100
    t0 = time.perf_counter()
    mgr.encode(BATCH_100)
    t_batch100 = time.perf_counter() - t0
    print(f"Batch-100 latency:{t_batch100*1000:.1f}ms ({t_batch100/100*1000:.1f}ms/text)")

    # Quality: similar pairs
    print("\nSimilarity quality:")
    for a, b in SIMILAR_PAIRS:
        ea, eb = mgr.encode([a])[0], mgr.encode([b])[0]
        sim = cosine_sim(ea, eb)
        print(f"  SIMILAR:    {sim:.4f}  {a[:50]}...")

    for a, b in DISSIMILAR_PAIRS:
        ea, eb = mgr.encode([a])[0], mgr.encode([b])[0]
        sim = cosine_sim(ea, eb)
        print(f"  DISSIMILAR: {sim:.4f}  {a[:50]}...")

    return {
        "model": "MiniLM-L6-v2",
        "load_time": load_time,
        "memory_mb": rss_after - rss_before,
        "dimensions": len(emb),
        "single_ms": np.mean(times) * 1000,
        "batch10_ms": t_batch10 * 1000,
        "batch100_ms": t_batch100 * 1000,
    }


def benchmark_gemma_gguf():
    """Benchmark EmbeddingGemma-300M Q4_0 GGUF via llama-server HTTP endpoint."""
    print("\n" + "=" * 70)
    print("MODEL: unsloth/embeddinggemma-300m-GGUF (Q4_0, llama-server HTTP)")
    print("=" * 70)

    import json as _json
    import urllib.request

    server_url = "http://localhost:8787"

    # Check server health
    try:
        req = urllib.request.Request(f"{server_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            health = _json.loads(resp.read())
            if health.get("status") != "ok":
                print(f"llama-server not healthy: {health}")
                return None
    except Exception as e:
        print(f"llama-server not running at {server_url}: {e}")
        print("Start it with: llama-server --model embeddinggemma-300m-Q4_0.gguf --port 8787 --embedding --ctx-size 512")
        return None

    def encode_texts(texts):
        """Encode texts via llama-server /embedding endpoint, return numpy array."""
        embeddings = []
        for t in texts:
            payload = _json.dumps({"content": t}).encode()
            req = urllib.request.Request(
                f"{server_url}/embedding",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            # Response: [{index: 0, embedding: [[768 floats]]}]
            emb = data[0]["embedding"]
            if isinstance(emb[0], list):
                emb = emb[0]
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    rss_before = get_rss_mb()

    # Warmup (server already has model loaded, this just warms HTTP path)
    t0 = time.perf_counter()
    warmup = encode_texts(["warmup"])
    load_time = time.perf_counter() - t0
    rss_after = get_rss_mb()

    print(f"First-call time:  {load_time:.3f}s (model already loaded in server)")
    print(f"Memory (RSS):     {rss_after - rss_before:.1f} MB (delta, client-side only)")

    # Dimensions
    emb = encode_texts([SINGLE_TEXT])
    if emb.ndim > 1:
        emb = emb[0]
    print(f"Dimensions:       {len(emb)}")

    # Single text latency (average of 10 runs)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        encode_texts([SINGLE_TEXT])
        times.append(time.perf_counter() - t0)
    print(f"Single latency:   {np.mean(times)*1000:.1f}ms (mean), {np.std(times)*1000:.1f}ms (std)")

    # Batch 10
    t0 = time.perf_counter()
    encode_texts(BATCH_10)
    t_batch10 = time.perf_counter() - t0
    print(f"Batch-10 latency: {t_batch10*1000:.1f}ms ({t_batch10/10*1000:.1f}ms/text)")

    # Batch 100
    t0 = time.perf_counter()
    encode_texts(BATCH_100)
    t_batch100 = time.perf_counter() - t0
    print(f"Batch-100 latency:{t_batch100*1000:.1f}ms ({t_batch100/100*1000:.1f}ms/text)")

    # Quality: similar pairs
    print("\nSimilarity quality:")
    for a, b in SIMILAR_PAIRS:
        ea = encode_texts([a])[0]
        eb = encode_texts([b])[0]
        sim = cosine_sim(ea, eb)
        print(f"  SIMILAR:    {sim:.4f}  {a[:50]}...")

    for a, b in DISSIMILAR_PAIRS:
        ea = encode_texts([a])[0]
        eb = encode_texts([b])[0]
        sim = cosine_sim(ea, eb)
        print(f"  DISSIMILAR: {sim:.4f}  {a[:50]}...")

    return {
        "model": "EmbeddingGemma-300M-Q4_0",
        "load_time": load_time,
        "memory_mb": rss_after - rss_before,
        "dimensions": len(emb),
        "single_ms": np.mean(times) * 1000,
        "batch10_ms": t_batch10 * 1000,
        "batch100_ms": t_batch100 * 1000,
    }


def print_comparison(r1, r2):
    """Print side-by-side comparison table."""
    if r1 is None or r2 is None:
        print("\nCannot compare — one model failed to load")
        return

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<22} {'MiniLM-L6-v2':>18} {'Gemma-300M-Q4':>18} {'Winner':>10}")
    print("-" * 70)

    rows = [
        ("Load time (s)", f"{r1['load_time']:.2f}", f"{r2['load_time']:.2f}",
         "MiniLM" if r1["load_time"] < r2["load_time"] else "Gemma"),
        ("Memory (MB)", f"{r1['memory_mb']:.0f}", f"{r2['memory_mb']:.0f}",
         "MiniLM" if r1["memory_mb"] < r2["memory_mb"] else "Gemma"),
        ("Dimensions", str(r1["dimensions"]), str(r2["dimensions"]), "-"),
        ("Single (ms)", f"{r1['single_ms']:.1f}", f"{r2['single_ms']:.1f}",
         "MiniLM" if r1["single_ms"] < r2["single_ms"] else "Gemma"),
        ("Batch-10 (ms)", f"{r1['batch10_ms']:.0f}", f"{r2['batch10_ms']:.0f}",
         "MiniLM" if r1["batch10_ms"] < r2["batch10_ms"] else "Gemma"),
        ("Batch-100 (ms)", f"{r1['batch100_ms']:.0f}", f"{r2['batch100_ms']:.0f}",
         "MiniLM" if r1["batch100_ms"] < r2["batch100_ms"] else "Gemma"),
        ("GGUF/quantized", "No", "Yes (Q4_0)", "-"),
        ("Model size (disk)", "~80MB", "~265MB", "MiniLM"),
    ]

    for label, v1, v2, winner in rows:
        print(f"{label:<22} {v1:>18} {v2:>18} {winner:>10}")


if __name__ == "__main__":
    print("Embedding Model Benchmark")
    print(f"CPU: {os.cpu_count()} cores")

    r1 = benchmark_minilm()

    # Force cleanup before loading second model
    gc.collect()

    r2 = benchmark_gemma_gguf()

    print_comparison(r1, r2)
