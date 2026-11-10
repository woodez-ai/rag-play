"""Text -> vector with Ollama, and cosine similarity by hand (numpy)."""

from typing import Literal

import numpy as np
import ollama

# Phase 1 result: mxbai beats nomic on raw descriptors (P@3 0.79 vs 0.21) and enriched text
# (0.92 vs 0.71/0.88). See notes/phase1-embeddings.md.
DEFAULT_MODEL = "mxbai-embed-large"

# Some embedding models are trained with task prefixes; leaving them off hurts retrieval.
# nomic-embed-text: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
PREFIXES: dict[str, dict[str, str]] = {
    "nomic-embed-text": {"query": "search_query: ", "document": "search_document: "},
    # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 (query side only)
    "mxbai-embed-large": {"query": "Represent this sentence for searching relevant passages: "},
}


def embed(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    kind: Literal["query", "document", "raw"] = "document",
) -> np.ndarray:
    """Embed texts as a (n, dim) float32 array. `kind` picks the model's task prefix."""
    prefix = "" if kind == "raw" else PREFIXES.get(model, {}).get(kind, "")
    resp = ollama.embed(model=model, input=[prefix + t for t in texts])
    return np.asarray(resp.embeddings, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n, d) and rows of b (m, d) -> (n, m).

    cos(a, b) = a·b / (|a| |b|): 1 = same direction, 0 = unrelated, -1 = opposite.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Indices and scores of the k most similar documents (exact search: compare against all)."""
    scores = cosine(query_vec, doc_vecs)[0]
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]
