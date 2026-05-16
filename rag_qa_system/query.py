"""RAG query: search vector DB → AI-generated answer with citations."""

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from anthropic import Anthropic
from anthropic.types import TextBlock
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DB_DIR = Path(__file__).parent.parent / "chroma_db"
TOP_K = 5  # Number of chunks to retrieve

_client = Anthropic(
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

RAG_SYSTEM_PROMPT = """\
You are a research assistant answering questions based on provided documents.

Rules:
1. Answer ONLY using the provided context chunks below
2. Cite sources using [Chunk N] notation (e.g. "According to the report [Chunk 2]...")
3. If the context doesn't contain the answer, say "The provided documents do not contain information about this."
4. Be concise. Directly answer the question, then provide supporting details.
5. If the context chunks contradict each other, note the contradiction.

--- CONTEXT CHUNKS ---
{context}
--- END CONTEXT ---
"""

MODEL = "deepseek-v4-pro[1m]"


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[str]  # Source file paths
    chunks_retrieved: int


def ask(question: str, collection: str = "default") -> QueryResult:
    """Search the vector DB and generate a cited answer."""
    # 1. Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        coll = client.get_collection(collection)
    except Exception:
        return QueryResult(
            question=question,
            answer="No documents ingested yet. Run `uv run python -m rag_qa_system.main ingest <path>` first.",
            sources=[],
            chunks_retrieved=0,
        )

    # 2. Search for relevant chunks
    results = coll.query(query_texts=[question], n_results=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return QueryResult(
            question=question,
            answer="No relevant chunks found in the documents.",
            sources=[],
            chunks_retrieved=0,
        )

    # 3. Format context with chunk labels
    context_parts = []
    sources = []
    seen_sources = set()

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_parts.append(f"[Chunk {i+1}] (Source: {meta.get('source', 'unknown')})\n{doc}")
        src = meta.get("source", "unknown")
        if src not in seen_sources:
            sources.append(src)
            seen_sources.add(src)

    context = "\n\n".join(context_parts)

    # 4. Send to AI
    response = _client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=RAG_SYSTEM_PROMPT.format(context=context),
        messages=[{"role": "user", "content": question}],
    )

    text = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            text += block.text

    return QueryResult(
        question=question,
        answer=text.strip(),
        sources=sources,
        chunks_retrieved=len(documents),
    )


def search_chunks(question: str, collection: str = "default") -> list[dict]:
    """Raw search: return chunks without AI summarization. Useful for debugging."""
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(collection)
    except Exception:
        return []

    results = coll.query(query_texts=[question], n_results=TOP_K)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "chunk_index": i,
            "source": metadatas[i].get("source", "?") if i < len(metadatas) else "?",
            "distance": distances[i] if i < len(distances) else 0,
            "text": documents[i][:300] + "..." if len(documents[i]) > 300 else documents[i],
        }
        for i in range(len(documents))
    ]
