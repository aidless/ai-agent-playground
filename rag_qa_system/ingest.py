"""
Document ingestion: load PDFs/TXTs → chunk (configurable strategy) → embed → ChromaDB.

Supports 3 chunking strategies (config.chunk_strategy):
  - fixed_size: character windows with overlap
  - sentence:    respects sentence boundaries (default, best for prose)
  - semantic:    splits on topic shifts via embedding similarity
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import create_chunker
from .config import RAGConfig


@dataclass
class IngestResult:
    files_processed: int
    chunks_created: int
    collection_name: str
    strategy: str


class DocumentIngester:
    """Loads documents, chunks them with configurable strategy, embeds with ChromaDB."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._db_dir = Path(config.db_dir)

    def ingest(self, path: str) -> IngestResult:
        """Ingest all PDF/TXT/MD files from a directory into ChromaDB."""
        dir_path = Path(path)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        files = self._find_documents(dir_path)
        if not files:
            print("No PDF, TXT, or MD files found.")
            return IngestResult(0, 0, self.config.collection, self.config.chunk_strategy)

        strategy = self.config.chunk_strategy
        chunker = create_chunker(
            strategy, self.config.chunk_size, self.config.chunk_overlap
        )

        print(f"Found {len(files)} document(s)")
        print(f"Chunking: {chunker.name()} (size={self.config.chunk_size}, "
              f"overlap={self.config.chunk_overlap})")
        print()

        client = chromadb.PersistentClient(
            path=str(self._db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            coll = client.get_collection(self.config.collection)
            print(f"Adding to '{self.config.collection}' "
                  f"({coll.count()} chunks already stored)\n")
        except Exception:
            coll = client.create_collection(self.config.collection)
            print(f"Created new collection '{self.config.collection}'\n")

        total = 0
        for fp in files:
            print(f"  Ingesting: {fp.name} ...", end=" ", flush=True)
            try:
                text = self._load_file(fp)
            except Exception as exc:
                print(f"SKIP ({exc})")
                continue

            chunks = chunker.split(text)
            if not chunks:
                print("SKIP (empty)")
                continue

            chunk_texts = [c.text for c in chunks]
            ids = [f"{fp.stem}_{c.chunk_index}" for c in chunks]
            metadatas = [
                {"source": str(fp), "chunk_index": c.chunk_index,
                 "start_char": c.start_char, "end_char": c.end_char}
                for c in chunks
            ]

            coll.add(ids=ids, documents=chunk_texts, metadatas=metadatas)
            total += len(chunks)
            print(f"{len(chunks)} chunks")

        print(f"\nDone. {total} chunks across {len(files)} file(s) "
              f"using [{chunker.name()}] strategy.")
        return IngestResult(len(files), total, self.config.collection, chunker.name())

    def stats(self) -> dict:
        """Return collection stats."""
        client = chromadb.PersistentClient(
            path=str(self._db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            coll = client.get_collection(self.config.collection)
            return {"name": self.config.collection, "chunks": coll.count()}
        except Exception:
            return {}

    # ---- Internal helpers ----

    @staticmethod
    def _find_documents(path: Path) -> list[Path]:
        files = []
        for ext in [".pdf", ".txt", ".md"]:
            files.extend(sorted(path.glob(f"*{ext}")))
        return files

    @staticmethod
    def _load_file(path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    parts.append(f"[Page {i+1}]\n{t.strip()}")
            return "\n\n".join(parts)
        return path.read_text(encoding="utf-8")
