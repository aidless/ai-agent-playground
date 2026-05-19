"""Knowledge Base API Routes — FastAPI router for research RAG."""

import logging
from fastapi import APIRouter, HTTPException

from .models import CollectRequest, QueryRequest
from .collector import PaperCollector
from .indexer import KnowledgeIndexer
from .query_engine import KnowledgeQueryEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

# Global instances — set by server.py on startup
collector: PaperCollector = None
indexer: KnowledgeIndexer = None
query_engine: KnowledgeQueryEngine = None


def init_knowledge_module(llm_client, llm_model="deepseek-chat"):
    """Called once at startup by server.py."""
    global collector, indexer, query_engine
    collector = PaperCollector()
    indexer = KnowledgeIndexer(collector=collector)
    query_engine = KnowledgeQueryEngine(
        indexer=indexer,
        collector=collector,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    logger.info("Knowledge module initialized")


@router.post("/collect")
async def collect_papers(request: CollectRequest):
    if not collector:
        raise HTTPException(status_code=503, detail="Knowledge module not initialized")
    result = collector.collect(max_papers=request.max_papers, topics=request.topics)
    if indexer:
        indexer.build_index()
    return {"success": True, "message": f"Collected {result['collected']} new papers", **result}


@router.post("/query")
async def rag_query(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Knowledge module not initialized")
    result = await query_engine.query(
        question=request.question,
        top_k=request.top_k,
        include_sources=request.include_sources,
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "source_count": len(result.get("sources", [])),
        "latency_seconds": result["latency_seconds"],
    }


@router.post("/search")
async def search_papers(query: str, top_k: int = 5):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Knowledge module not initialized")
    return query_engine.search(query, top_k=top_k)


@router.get("/status")
async def get_status():
    coll_status = collector.get_status() if collector else {}
    idx_status = indexer.get_status() if indexer else {}
    return {
        "total_papers": coll_status.get("total_papers", 0),
        "total_chunks": idx_status.get("total_chunks", 0),
        "last_updated": "",
        "index_size_mb": round(idx_status.get("index_size_mb", 0), 2),
        "embedding_model": idx_status.get("embedding_model", ""),
        "llm_model": query_engine.llm_model if query_engine else "",
        "gpu_available": idx_status.get("gpu_available", False),
    }


@router.post("/reindex")
async def rebuild_index(force: bool = False):
    if not indexer:
        raise HTTPException(status_code=503, detail="Knowledge module not initialized")
    result = indexer.build_index(force_rebuild=force)
    return {"success": True, **result}
