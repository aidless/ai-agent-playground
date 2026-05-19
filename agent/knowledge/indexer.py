"""Vector Index Builder — ChromaDB + sentence-transformers for paper embeddings."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VECTOR_DIR = Path(__file__).resolve().parent.parent.parent / "memory" / "knowledge" / "chroma"
COLLECTION_NAME = "agent_knowledge"


class KnowledgeIndexer:
    """Builds and maintains ChromaDB vector index over research papers."""

    def __init__(self, collector=None):
        self.collector = collector
        self._chroma_client = None
        self._collection = None
        self._embed_fn = None
        self._built = False

    def build_index(self, force_rebuild: bool = False) -> dict:
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))

            if force_rebuild:
                try:
                    self._chroma_client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass

            try:
                self._collection = self._chroma_client.get_collection(COLLECTION_NAME)
                logger.info("Collection exists: %d documents", self._collection.count())
            except Exception:
                self._collection = self._chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "AI Agent research papers"},
                )

            # Load embedding model
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_fn = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Embedding model loaded: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not available, using keyword-only search")
                self._embed_fn = None

            if not self.collector or not self.collector._cache:
                logger.warning("No papers to index")
                return {"error": "No papers found. Run collection first."}

            papers = list(self.collector._cache.values())
            batch_size = 20
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i + batch_size]
                ids = [p.arxiv_id for p in batch]
                texts = [p.chunk_text for p in batch]
                metadatas = [{"title": p.title, "authors": ", ".join(p.authors[:3]),
                             "published": p.published} for p in batch]

                if self._embed_fn:
                    embeddings = self._embed_fn.encode(texts).tolist()
                    self._collection.add(ids=ids, embeddings=embeddings,
                                        documents=texts, metadatas=metadatas)
                else:
                    self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

            self._built = True
            logger.info("Indexed %d papers", len(papers))
            return {"documents": len(papers), "chunks": self._collection.count()}
        except ImportError:
            return {"error": "chromadb not installed"}
        except Exception as e:
            logger.warning("Index build failed: %s", e)
            return {"error": str(e)}

    def get_status(self) -> dict:
        count = 0
        try:
            if self._collection:
                count = self._collection.count()
            elif self._chroma_client:
                try:
                    coll = self._chroma_client.get_collection(COLLECTION_NAME)
                    count = coll.count()
                except Exception:
                    pass
        except Exception:
            pass

        size = 0.0
        if VECTOR_DIR.exists():
            for f in VECTOR_DIR.rglob("*"):
                if f.is_file():
                    size += f.stat().st_size

        return {
            "total_chunks": count,
            "index_size_mb": size / (1024 * 1024),
            "embedding_model": "all-MiniLM-L6-v2",
            "gpu_available": False,
        }

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def collection(self):
        return self._collection

    @property
    def embed_fn(self):
        return self._embed_fn
