"""
RAG query: hybrid search → AI-generated answer with citations.

Supports two search modes:
  - vector:   pure dense vector search (ChromaDB)
  - hybrid:   dense + sparse (BM25) fused via Reciprocal Rank Fusion
"""

from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_agent_playground.llm import LLMClient, get_client

from .config import RAGConfig
from .hybrid_search import HybridSearcher
from .reranker import create_reranker


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks_retrieved: int = 0
    search_mode: str = "vector"


class RAGQuerier:
    """Searches ChromaDB (or hybrid) and generates cited answers via LLM."""

    def __init__(self, config: RAGConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm or get_client()
        self._db_dir = Path(config.db_dir)
        self._hybrid: HybridSearcher | None = None
        self._reranker = None  # lazy-loaded

    def _get_hybrid(self) -> HybridSearcher:
        if self._hybrid is None:
            self._hybrid = HybridSearcher(
                db_dir=str(self._db_dir),
                collection=self.config.collection,
                vector_weight=self.config.vector_weight,
            )
        return self._hybrid

    def ask(self, question: str) -> QueryResult:
        """Search + generate cited answer."""
        if self.config.search_mode == "hybrid":
            return self._ask_hybrid(question)
        return self._ask_vector(question)

    def _ask_vector(self, question: str) -> QueryResult:
        """Pure vector search — original path."""
        client = chromadb.PersistentClient(
            path=str(self._db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            coll = client.get_collection(self.config.collection)
        except Exception:
            return QueryResult(
                question=question,
                answer="No documents ingested yet. "
                       "Run `uv run python -m rag_qa_system.main ingest <path>` first.",
            )

        results = coll.query(
            query_texts=[question],
            n_results=self.config.oversampling_k,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return self._generate_answer(question, documents, metadatas, "vector")

    def _rerank(
        self,
        question: str,
        documents: list[str],
        metadatas: list[dict],
        scores: list[float] | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Re-rank documents using a cross-encoder for higher precision."""
        if self._reranker is None:
            self._reranker = create_reranker()
        if self._reranker is None:
            return documents, metadatas

        candidates = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            candidates.append({
                "text": doc,
                "score": scores[i] if scores and i < len(scores) else 0.5,
                "meta": meta,
            })

        reranked = self._reranker.rerank(
            question, candidates, top_k=self.config.top_k
        )
        docs = [r.text for r in reranked]
        metas = [candidates[r.chunk_index]["meta"] for r in reranked]
        return docs, metas

    def _ask_hybrid(self, question: str) -> QueryResult:
        """Hybrid search: vector + BM25 → RRF fusion."""
        searcher = self._get_hybrid()
        hybrid_results = searcher.search(
            question, top_k=self.config.oversampling_k
        )

        if not hybrid_results:
            return QueryResult(
                question=question,
                answer="No relevant chunks found in the documents.",
                search_mode="hybrid",
            )

        documents = [r.text for r in hybrid_results]
        metadatas = [{"source": r.source} for r in hybrid_results]

        return self._generate_answer(question, documents, metadatas, "hybrid")

    def _generate_answer(
        self,
        question: str,
        documents: list[str],
        metadatas: list[dict],
        search_mode: str,
    ) -> QueryResult:
        """Build context + generate cited answer via LLM."""
        if not documents:
            return QueryResult(
                question=question,
                answer="No relevant chunks found in the documents.",
                search_mode=search_mode,
            )

        # Re-rank with cross-encoder for higher precision
        docs, metas = self._rerank(question, documents, metadatas)
        docs = docs[:self.config.top_k]
        metas = metas[:self.config.top_k]

        context_parts = []
        sources = []
        seen = set()

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            context_parts.append(
                f"[Chunk {i + 1}] (Source: {meta.get('source', 'unknown')})\n{doc}"
            )
            src = meta.get("source", "unknown")
            if src not in seen:
                sources.append(src)
                seen.add(src)

        context = "\n\n".join(context_parts)
        system_prompt = self.config.system_prompt.format(context=context)

        answer = self.llm.send(
            messages=[{"role": "user", "content": question}],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
        )

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            chunks_retrieved=len(docs),
            search_mode=search_mode,
        )

    def search(self, question: str) -> list[dict]:
        """Raw chunk search — useful for debugging retrieval quality."""
        client = chromadb.PersistentClient(
            path=str(self._db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            coll = client.get_collection(self.config.collection)
        except Exception:
            return []

        results = coll.query(
            query_texts=[question], n_results=self.config.top_k
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {
                "chunk_index": i,
                "source": metas[i].get("source", "?"),
                "distance": dists[i],
                "text": docs[i][:300] + "..." if len(docs[i]) > 300 else docs[i],
            }
            for i in range(len(docs))
        ]
