"""
Hybrid Search — combines dense (vector) and sparse (BM25) retrieval.

Why hybrid?
  - Vector search is great for semantic similarity ("car" ≈ "automobile")
    but misses exact keyword matches ("AK-47" won't match "rifle" well)
  - BM25 is great for exact keywords and rare terms
    but misses paraphrases and synonyms

Reciprocal Rank Fusion (RRF) merges both result lists without needing
to normalize scores across fundamentally different score distributions.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods", SIGIR 2009.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


# ============================================================
#  BM25 Index (sparse / keyword search)
# ============================================================


class BM25Index:
    """Minimal BM25 implementation. No external dependencies.

    BM25 = "Best Match 25", the classic IR scoring function.
    It scores documents by term frequency (TF) dampened by document length,
    multiplied by inverse document frequency (IDF).

    Formula: score(D, Q) = Σ IDF(qi) * (TF(qi, D) * (k1 + 1))
                              / (TF(qi, D) + k1 * (1 - b + b * |D|/avgDL))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []
        self._term_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}

    def index(self, documents: list[str]):
        """Build BM25 index from a list of document texts."""
        self._docs = documents
        self._term_freqs = []
        self._doc_lengths = []

        total_docs = len(documents)
        doc_freq = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self._doc_lengths.append(len(tokens))
            self._term_freqs.append(Counter(tokens))
            for term in set(tokens):
                doc_freq[term] += 1

        self._avg_dl = (
            sum(self._doc_lengths) / total_docs if total_docs else 1.0
        )

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log(
                (total_docs - df + 0.5) / (df + 0.5) + 1
            )

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return list of (doc_index, bm25_score), sorted by score desc."""
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, (tf, dl) in enumerate(zip(self._term_freqs, self._doc_lengths)):
            score = 0.0
            for token in query_tokens:
                if token not in self._idf:
                    continue
                f = tf.get(token, 0)
                if f == 0:
                    continue
                idf = self._idf[token]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (
                    1 - self.b + self.b * dl / self._avg_dl
                )
                score += idf * numerator / denominator

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple multilingual tokenizer (handles Chinese + English)."""
        # Split on word boundaries, keep Chinese chars as individual tokens
        return re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', text.lower())


# ============================================================
#  Reciprocal Rank Fusion (RRF)
# ============================================================


def reciprocal_rank_fusion(
    vector_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
    vector_weight: float = 0.6,
) -> list[tuple[int, float]]:
    """Merge two ranked lists using weighted Reciprocal Rank Fusion.

    RRF score(doc) = Σ w * 1/(k + rank(doc, list))

    k=60 is the standard value from the RRF paper.
    Higher k → smoother blending, less sensitive to rank differences.

    Args:
        vector_results: [(doc_idx, score), ...] from vector search
        bm25_results: [(doc_idx, score), ...] from BM25
        k: RRF smoothing constant
        vector_weight: weight for vector results (bm25_weight = 1 - vector_weight)
    """
    scores: dict[int, float] = {}

    for rank, (doc_idx, _) in enumerate(vector_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + vector_weight / (k + rank + 1)

    for rank, (doc_idx, _) in enumerate(bm25_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + (1 - vector_weight) / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# ============================================================
#  Hybrid Searcher — combines ChromaDB + BM25
# ============================================================


@dataclass
class HybridSearchResult:
    chunk_index: int
    doc_id: str
    text: str
    source: str
    score: float
    vector_score: float
    bm25_score: float


class HybridSearcher:
    """Searches with both ChromaDB (dense) and BM25 (sparse), fuses via RRF."""

    def __init__(
        self,
        db_dir: str = "chroma_db",
        collection: str = "default",
        vector_weight: float = 0.6,
    ):
        self.db_dir = Path(db_dir)
        self.collection = collection
        self.vector_weight = vector_weight
        self._bm25: BM25Index | None = None
        self._id_map: list[str] = []  # index → chromadb id

    def _ensure_bm25(self):
        """Lazy-load: build BM25 index from all chunks in ChromaDB."""
        if self._bm25 is not None:
            return

        client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            coll = client.get_collection(self.collection)
        except Exception:
            self._bm25 = BM25Index()
            return

        # Fetch all documents (for small-to-medium collections)
        # For large collections, this should be batched
        results = coll.get()
        docs = results.get("documents", [])
        ids = results.get("ids", [])

        self._bm25 = BM25Index()
        if docs:
            self._bm25.index(docs)
            self._id_map = ids

    def search(
        self, query: str, top_k: int = 5
    ) -> list[HybridSearchResult]:
        """Hybrid search: vector + BM25 → RRF fusion."""
        self._ensure_bm25()

        client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            coll = client.get_collection(self.collection)
        except Exception:
            return []

        # Vector search
        K = max(top_k * 3, 15)  # oversized retrieval: fetch more, re-rank
        vec_results = coll.query(query_texts=[query], n_results=K)
        vec_docs = vec_results.get("documents", [[]])[0]
        vec_ids = vec_results.get("ids", [[]])[0]
        vec_metas = vec_results.get("metadatas", [[]])[0]
        vec_dists = vec_results.get("distances", [[]])[0]

        # BM25 search
        bm25_results = self._bm25.search(query, top_k=K)

        # Map vector results to indices (use position as proxy index)
        # BM25 uses internal indices, vector uses ChromaDB indices
        # We need to align them. Strategy: use ChromaDB id as join key
        vec_idx_map = {vid: i for i, vid in enumerate(vec_ids)}

        # Convert BM25 to (position, score) aligned on chromadb id
        bm25_aligned = []
        for bm25_idx, bm25_score in bm25_results:
            if bm25_idx < len(self._id_map):
                chroma_id = self._id_map[bm25_idx]
                if chroma_id in vec_idx_map:
                    bm25_aligned.append((vec_idx_map[chroma_id], bm25_score))

        # Vector results already indexed by position
        vec_ranked = list(enumerate(vec_dists))

        # RRF fusion
        fused = reciprocal_rank_fusion(
            vec_ranked, bm25_aligned, vector_weight=self.vector_weight
        )

        # Build results
        results = []
        for fused_idx, fused_score in fused[:top_k]:
            if fused_idx < len(vec_docs):
                results.append(HybridSearchResult(
                    chunk_index=fused_idx,
                    doc_id=vec_ids[fused_idx] if fused_idx < len(vec_ids) else "?",
                    text=vec_docs[fused_idx],
                    source=vec_metas[fused_idx].get("source", "?")
                    if fused_idx < len(vec_metas) else "?",
                    score=fused_score,
                    vector_score=1.0 - vec_dists[fused_idx]
                    if fused_idx < len(vec_dists) else 0.0,
                    bm25_score=bm25_aligned[fused_idx][1]
                    if fused_idx < len(bm25_aligned) else 0.0,
                ))
        return results
