"""Document ingestion: load PDFs/TXTs → chunk → embed → store in ChromaDB.

Like a tokenizer: raw documents → structured vectors ready for retrieval.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import RAGConfig


@dataclass
class IngestResult:
    files_processed: int
    chunks_created: int
    collection_name: str


class DocumentIngester:
    """Loads documents, chunks them, embeds with ChromaDB's built-in model.

    ChromaDB uses all-MiniLM-L6-v2 (ONNX) by default — no API key, runs locally.
    """

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
            return IngestResult(0, 0, self.config.collection)

        print(f"Found {len(files)} document(s)\n")

        client = chromadb.PersistentClient(
            path=str(self._db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection (ChromaDB's built-in all-MiniLM-L6-v2 for embeddings)
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

            chunks = self._chunk_text(text)
            if not chunks:
                print("SKIP (empty)")
                continue

            ids = [f"{fp.stem}_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": str(fp), "chunk_index": i}
                for i in range(len(chunks))
            ]
            coll.add(ids=ids, documents=chunks, metadatas=metadatas)
            total += len(chunks)
            print(f"{len(chunks)} chunks")

        print(f"\nDone. {total} chunks across {len(files)} file(s).")
        return IngestResult(len(files), total, self.config.collection)

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

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) < self.config.chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                    current = current[-self.config.chunk_overlap:] + "\n\n" + para
                else:
                    for i in range(0, len(para), self.config.chunk_size - self.config.chunk_overlap):
                        chunk = para[i:i + self.config.chunk_size]
                        if len(chunk) > 50:
                            chunks.append(chunk)
                    current = ""

        if current and len(current) > 20:
            chunks.append(current)
        return chunks
