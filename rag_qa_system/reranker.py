"""
Cross-Encoder Re-ranker — boost retrieval precision by re-scoring candidates.

Vector search (bi-encoder) is fast but imprecise — it compares query and
document embeddings independently. Cross-encoders process the query-document
PAIR together, producing much more accurate relevance scores.

The standard production pattern:
  1. Bi-encoder retrieves top-K (fast, approximate) — e.g. K=100
  2. Cross-encoder re-scores top-K (slower, precise) — re-rank to top-k=5

Supported backends:
  - BGE-Reranker (BAAI/bge-reranker-v2-m3) — recommended for Chinese+English
  - sentence-transformers CrossEncoder — any HF cross-encoder model
  - API-based (Cohere Rerank, Jina Reranker) — for serverless

Reference: Nogueira et al., "Passage Re-ranking with BERT", arXiv:1901.04085
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RerankResult:
    chunk_index: int
    text: str
    original_score: float  # from first-stage retrieval
    rerank_score: float  # from cross-encoder
    combined_score: float  # weighted combination


class Reranker:
    """Cross-encoder re-ranking for retrieval results.

    Usage:
        reranker = Reranker("BAAI/bge-reranker-v2-m3")
        results = [{"text": "...", "score": 0.85}, ...]
        reranked = reranker.rerank("query", results, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_gpu: bool = False,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._use_gpu = use_gpu

    def _lazy_load(self):
        """Lazy-load the cross-encoder model (can be 1-2GB)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            device = "cuda" if self._use_gpu else "cpu"
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: uv add sentence-transformers"
            )
        except Exception as e:
            # If model download fails, provide a clear fallback message
            raise RuntimeError(
                f"Failed to load reranker model '{self.model_name}': {e}\n"
                "Options:\n"
                "  1. Set reranker_model='BAAI/bge-reranker-base' (lighter, ~1GB)\n"
                "  2. Use API-based: Cohere Rerank or Jina Reranker\n"
                "  3. Set search_mode='vector' to skip re-ranking"
            )

    def is_available(self) -> bool:
        """Check if re-ranker can be loaded without actually loading it."""
        try:
            import importlib
            importlib.import_module("sentence_transformers")
            return True
        except ImportError:
            return False

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        weight_original: float = 0.3,
    ) -> list[RerankResult]:
        """Re-rank candidate chunks using cross-encoder scoring.

        Args:
            query: the user's question
            candidates: list of dicts with 'text', 'score', and optional metadata
            top_k: how many results to return after re-ranking
            weight_original: weight for original score (0-1).
                             0 = pure cross-encoder, 1 = pure original

        Returns:
            Re-ranked list of RerankResult, sorted by combined_score desc
        """
        if not candidates:
            return []

        self._lazy_load()

        # Build query-document pairs for cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Cross-encoder scores (higher = more relevant)
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Normalize cross-encoder scores to 0-1 using softmax
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        normalized = self._softmax(scores_list)

        # Combine with original scores
        results = []
        for i, candidate in enumerate(candidates):
            orig = candidate.get("score", 0.5)
            cross = normalized[i] if i < len(normalized) else 0.0
            combined = weight_original * orig + (1 - weight_original) * cross

            results.append(RerankResult(
                chunk_index=i,
                text=candidate["text"],
                original_score=orig,
                rerank_score=cross,
                combined_score=combined,
            ))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        """Softmax normalization — makes scores sum to 1."""
        if not scores:
            return []
        # Subtract max for numerical stability
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        return [e / total for e in exps] if total > 0 else [1.0 / len(scores)] * len(scores)


def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    use_gpu: bool = False,
) -> Reranker | None:
    """Factory: create a reranker if available, None if dependencies missing."""
    try:
        reranker = Reranker(model_name=model_name, use_gpu=use_gpu)
        if reranker.is_available():
            return reranker
    except Exception:
        pass
    return None
