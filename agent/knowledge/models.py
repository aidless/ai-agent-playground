"""Data models for knowledge base module."""

from pydantic import BaseModel, Field
from typing import Optional


class KnowledgeStatus(BaseModel):
    total_papers: int = 0
    total_chunks: int = 0
    last_updated: str = ""
    index_size_mb: float = 0.0
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "deepseek-chat"
    gpu_available: bool = False


class SearchResult(BaseModel):
    id: str
    title: str = ""
    authors: str = ""
    abstract: str = ""
    relevance_score: float = 0.0
    pdf_url: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchResult] = []
    tokens_used: int = 0
    latency_seconds: float = 0.0


class CollectRequest(BaseModel):
    max_papers: int = Field(50, ge=1, le=200)
    topics: Optional[list[str]] = None


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True
