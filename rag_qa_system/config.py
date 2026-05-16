"""RAG Q&A System config — like BertConfig, declare params with defaults."""

from dataclasses import dataclass, field
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class RAGConfig(BaseAgentConfig):
    """Configuration for RAG Q&A System.

    Like BertConfig declaring vocab_size, hidden_size, etc.
    — we declare chunking, retrieval, and generation params.
    """

    agent_type: ClassVar[str] = "rag-qa"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a research assistant answering questions based on provided documents.\n\n"
        "Rules:\n"
        "1. Answer ONLY using the provided context chunks below\n"
        '2. Cite sources using [Chunk N] notation (e.g. "According to [Chunk 2]...")\n'
        '3. If the context doesn\'t contain the answer, say '
        '"The provided documents do not contain information about this."\n'
        "4. Be concise. Directly answer the question, then provide supporting details.\n"
        "5. If the context chunks contradict each other, note the contradiction.\n\n"
        "--- CONTEXT CHUNKS ---\n"
        "{context}\n"
        "--- END CONTEXT ---"
    )

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5

    # Storage
    db_dir: str = "chroma_db"
    collection: str = "default"
