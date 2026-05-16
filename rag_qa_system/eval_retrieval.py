"""
Retrieval quality evaluation — because "looks about right" isn't engineering.

Metrics (the standard IR eval suite):
  - recall@k       — what fraction of relevant docs did we find in top-k?
  - precision@k    — what fraction of top-k results are actually relevant?
  - MRR            — mean reciprocal rank (how early is the first hit?)
  - NDCG@k         — position-weighted relevance (higher = better ranking)

Usage:
  uv run python -m rag_qa_system.eval_retrieval
  uv run python -m rag_qa_system.eval_retrieval --compare  # compare all strategies
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import RAGConfig
from .chunking import create_chunker
from .hybrid_search import HybridSearcher


# ============================================================
#  Test dataset
# ============================================================


@dataclass
class RetrievalQuery:
    question: str
    relevant_texts: list[str]  # substrings that should match relevant chunks


@dataclass
class RetrievalTestSet:
    name: str
    queries: list[RetrievalQuery]


# ============================================================
#  Metrics
# ============================================================


def recall_at_k(relevant: set[int], retrieved: list[int], k: int = 5) -> float:
    """What fraction of relevant documents did we find in top-k?"""
    if not relevant:
        return 1.0
    found = len(relevant & set(retrieved[:k]))
    return found / len(relevant)


def precision_at_k(relevant: set[int], retrieved: list[int], k: int = 5) -> float:
    """What fraction of top-k results are relevant?"""
    if not retrieved[:k]:
        return 0.0
    found = len(relevant & set(retrieved[:k]))
    return found / min(k, len(retrieved))


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant hit.

    MRR = 1.0 means first result is always relevant.
    MRR = 0.1 means best hit is around position 10.
    """
    for i, doc_idx in enumerate(retrieved):
        if doc_idx in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain — position-weighted relevance.

    Higher rank positions have exponentially decaying weight.
    NDCG = DCG / IDCG (normalized to ideal ranking).
    """
    if not relevant:
        return 1.0

    # Binary relevance: 1 if in relevant set, 0 otherwise
    dcg = 0.0
    for i, doc_idx in enumerate(retrieved[:k]):
        rel = 1.0 if doc_idx in relevant else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: all relevant docs at the top
    idcg = 0.0
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalMetrics:
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float

    def summary(self) -> str:
        return (
            f"R@3={self.recall_at_3:.2f} R@5={self.recall_at_5:.2f} "
            f"P@5={self.precision_at_5:.2f} MRR={self.mrr:.2f} NDCG@5={self.ndcg_at_5:.2f}"
        )


# ============================================================
#  Benchmark runner
# ============================================================


def _load_testset(path: str | None = None) -> RetrievalTestSet:
    """Load test queries from JSON, or generate a default testset from ChromaDB."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return RetrievalTestSet(
            name=data.get("name", "custom"),
            queries=[RetrievalQuery(**q) for q in data.get("queries", [])],
        )

    # Default: generate queries from ingested documents
    return _generate_default_testset()


def _generate_default_testset() -> RetrievalTestSet:
    """Auto-generate test queries from ChromaDB chunks.

    For each unique source document, create a query containing key terms.
    This is a lightweight approach — in production, you'd manually label.
    """
    config = RAGConfig()
    client = chromadb.PersistentClient(
        path=str(config.db_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(config.collection)
    except Exception:
        return RetrievalTestSet(name="empty", queries=[])

    data = coll.get()
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not docs:
        return RetrievalTestSet(name="empty", queries=[])

    # Group chunks by source
    by_source: dict[str, list[tuple[int, str]]] = {}
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        src = meta.get("source", "unknown")
        by_source.setdefault(src, []).append((i, doc))

    queries = []
    for src, chunks in by_source.items():
        if len(chunks) < 2:
            continue
        # Take a chunk and extract query terms from it
        for idx, (chunk_i, text) in enumerate(chunks[:min(3, len(chunks))]):
            words = text.split()[:10]
            if len(words) < 3:
                continue
            query_terms = " ".join(words[:6])
            # Relevant: this chunk and adjacent chunks
            relevant = [c[0] for c in chunks[max(0, idx - 1):idx + 2]]
            queries.append(RetrievalQuery(
                question=f"What does the document say about {query_terms}...?",
                relevant_texts=[docs[ri][:100] for ri in relevant],
            ))
        if len(queries) >= 10:
            break

    return RetrievalTestSet(
        name=f"auto-generated ({len(queries)} queries)",
        queries=queries[:10],
    )


def _find_relevant_indices(relevant_texts: list[str], all_docs: list[str]) -> set[int]:
    """Match relevant_texts to document indices via substring matching."""
    relevant = set()
    for rt in relevant_texts:
        for i, doc in enumerate(all_docs):
            if rt[:60] in doc:
                relevant.add(i)
    return relevant


def evaluate_retrieval(
    searcher_mode: str = "vector",
    config: RAGConfig | None = None,
    testset_path: str | None = None,
) -> dict[str, Any]:
    """Run full retrieval evaluation.

    Args:
        searcher_mode: "vector" | "hybrid"
        config: RAG config
        testset_path: path to test queries JSON
    """
    if config is None:
        config = RAGConfig()

    testset = _load_testset(testset_path)
    if not testset.queries:
        print("No test queries available. Ingest documents first.")
        return {}

    # Load ChromaDB data
    client = chromadb.PersistentClient(
        path=str(config.db_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(config.collection)
    except Exception:
        print("No collection found. Run ingestion first.")
        return {}

    all_data = coll.get()
    all_docs = all_data.get("documents", [])

    # Setup searchers
    hybrid_searcher = None
    if searcher_mode == "hybrid":
        hybrid_searcher = HybridSearcher(
            db_dir=config.db_dir,
            collection=config.collection,
        )

    per_query_metrics = []
    for q in testset.queries:
        relevant = _find_relevant_indices(q.relevant_texts, all_docs)

        if searcher_mode == "hybrid" and hybrid_searcher:
            results = hybrid_searcher.search(q.question, top_k=5)
            retrieved = [r.chunk_index for r in results if r.chunk_index < len(all_docs)]
        else:
            vec_results = coll.query(query_texts=[q.question], n_results=5)
            retrieved = list(range(len(vec_results.get("documents", [[]])[0])))

        per_query_metrics.append({
            "question": q.question,
            "relevant_count": len(relevant),
            "retrieved_count": len(retrieved),
            "recall@5": recall_at_k(relevant, retrieved, k=5),
            "mrr": mrr(relevant, retrieved),
            "ndcg@5": ndcg_at_k(relevant, retrieved, k=5),
        })

    if not per_query_metrics:
        return {}

    # Aggregate
    avg = EvalMetrics(
        recall_at_3=sum(m["recall@5"] for m in per_query_metrics) / len(per_query_metrics),
        recall_at_5=sum(m["recall@5"] for m in per_query_metrics) / len(per_query_metrics),
        precision_at_5=0.0,  # requires manual labels for true precision
        mrr=sum(m["mrr"] for m in per_query_metrics) / len(per_query_metrics),
        ndcg_at_5=sum(m["ndcg@5"] for m in per_query_metrics) / len(per_query_metrics),
    )

    return {
        "searcher_mode": searcher_mode,
        "num_queries": len(per_query_metrics),
        "avg_metrics": avg,
        "per_query": per_query_metrics,
    }


# ============================================================
#  Compare all strategies
# ============================================================


def compare_strategies(config: RAGConfig | None = None) -> list[dict]:
    """Run retrieval eval with each search mode, compare results.

    Prints a comparison table suitable for decision-making.
    """
    if config is None:
        config = RAGConfig()

    results = []
    for mode in ["vector", "hybrid"]:
        print(f"\n  Evaluating: {mode} ...", end=" ", flush=True)
        result = evaluate_retrieval(searcher_mode=mode, config=config)
        if result:
            results.append(result)
            print(result["avg_metrics"].summary())
        else:
            print("SKIPPED (no data)")

    return results


# ============================================================
#  CLI
# ============================================================


def main():
    import sys

    config = RAGConfig()

    if "--compare" in sys.argv:
        print("=" * 60)
        print("  Retrieval Strategy Comparison")
        print("=" * 60)
        results = compare_strategies(config)

        if len(results) >= 2:
            print(f"\n{'=' * 60}")
            print("  Verdict:")
            vec = results[0]["avg_metrics"]
            hyb = results[1]["avg_metrics"]
            print(f"  Vector only  → {vec.summary()}")
            print(f"  Hybrid (RRF) → {hyb.summary()}")

            if hyb.mrr > vec.mrr:
                print(f"  → Hybrid search wins on MRR (+{(hyb.mrr - vec.mrr):.2f})")
            else:
                print(f"  → Vector search sufficient for this dataset")
    else:
        print("=" * 60)
        print("  Retrieval Quality Evaluation")
        print("=" * 60)
        result = evaluate_retrieval(searcher_mode="vector", config=config)
        if result:
            print(f"\n  {result['num_queries']} queries evaluated")
            print(f"  {result['avg_metrics'].summary()}")

            print(f"\n  Per-query breakdown:")
            for q in result["per_query"]:
                print(f"    [{q['recall@5']:.2f} R@5 | {q['mrr']:.2f} MRR] {q['question'][:80]}...")


if __name__ == "__main__":
    main()
