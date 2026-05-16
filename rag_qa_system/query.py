"""RAG query: vector search → AI-generated answer with citations.

Like model.generate(): the core inference step — search + generate.
"""

from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_agent_playground.llm import LLMClient, get_client

from .config import RAGConfig


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks_retrieved: int = 0


class RAGQuerier:
    """Searches ChromaDB and generates cited answers via the LLM.

    Like a decoder: context + question → cited answer.
    """

    def __init__(self, config: RAGConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm or get_client()
        self._db_dir = Path(config.db_dir)

    def ask(self, question: str) -> QueryResult:
        """Search vector DB, generate cited answer."""
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

        # Vector search
        results = coll.query(query_texts=[question], n_results=self.config.top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return QueryResult(
                question=question,
                answer="No relevant chunks found in the documents.",
            )

        # Build context with chunk labels
        context_parts = []
        sources = []
        seen = set()

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            context_parts.append(
                f"[Chunk {i+1}] (Source: {meta.get('source', 'unknown')})\n{doc}"
            )
            src = meta.get("source", "unknown")
            if src not in seen:
                sources.append(src)
                seen.add(src)

        context = "\n\n".join(context_parts)

        # Generate cited answer
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
            chunks_retrieved=len(documents),
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

        results = coll.query(query_texts=[question], n_results=self.config.top_k)
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
