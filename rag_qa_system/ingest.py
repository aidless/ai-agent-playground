"""Document ingestion: load PDFs/TXTs → chunk → embed → store in ChromaDB."""

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


# Where the vector database lives
DB_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between adjacent chunks


@dataclass
class IngestResult:
    files_processed: int
    chunks_created: int
    collection_name: str


def ingest_directory(path: str, collection: str = "default") -> IngestResult:
    """Ingest all PDF and TXT files from a directory into ChromaDB."""
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    files = _find_documents(dir_path)
    if not files:
        print("No PDF or TXT files found.")
        return IngestResult(0, 0, collection)

    print(f"Found {len(files)} document(s)\n")

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Get or create collection. ChromaDB uses its built-in all-MiniLM-L6-v2
    # embedding function by default — no API key needed, runs locally.
    try:
        coll = client.get_collection(collection)
        print(f"Adding to existing collection '{collection}' "
              f"({coll.count()} chunks already stored)\n")
    except Exception:
        coll = client.create_collection(collection)
        print(f"Created new collection '{collection}'\n")

    total_chunks = 0
    for file_path in files:
        print(f"  Ingesting: {file_path.name} ...", end=" ", flush=True)
        try:
            text = _load_file(file_path)
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

        chunks = _chunk_text(text)
        if not chunks:
            print("SKIP (empty)")
            continue

        ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": str(file_path), "chunk_index": i}
            for i in range(len(chunks))
        ]

        coll.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"{len(chunks)} chunks")

    print(f"\nDone. {total_chunks} chunks total across {len(files)} file(s).")
    return IngestResult(
        files_processed=len(files),
        chunks_created=total_chunks,
        collection_name=collection,
    )


def _find_documents(dir_path: Path) -> list[Path]:
    """Find all PDF and TXT files, sorted by name."""
    files = []
    for ext in [".pdf", ".txt", ".md"]:
        files.extend(sorted(dir_path.glob(f"*{ext}")))
    return files


def _load_file(path: Path) -> str:
    """Load text from a file. PDFs go through pypdf extraction."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                parts.append(f"[Page {i+1}]\n{page_text.strip()}")
        return "\n\n".join(parts)
    else:
        return path.read_text(encoding="utf-8")


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Keep overlap for continuity
                current = current[-CHUNK_OVERLAP:] + "\n\n" + para
            else:
                # Single paragraph is longer than chunk size — split it
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = para[i:i + CHUNK_SIZE]
                    if len(chunk) > 50:
                        chunks.append(chunk)
                current = ""

    if current and len(current) > 20:
        chunks.append(current)

    return chunks


def collection_stats(collection: str = "default") -> dict:
    """Return stats for a collection, or empty dict if it doesn't exist."""
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(collection)
        return {"name": collection, "chunks": coll.count()}
    except Exception:
        return {}
