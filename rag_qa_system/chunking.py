"""
Chunking strategies — how to split documents into retrievable pieces.

Chunking is the most underrated lever in RAG performance.
A bad chunking strategy silently destroys retrieval quality:
  - Too small → loses context, embeddings are noisy
  - Too large → dilutes relevance signal, wastes context window
  - Wrong boundaries → splits a key concept across two chunks

Three strategies, from simple to smart:
  1. FixedSizeChunker  — character-count windows (fast, predictable)
  2. SentenceChunker   — respects linguistic boundaries (cleaner chunks)
  3. SemanticChunker   — splits on topic shifts via embedding similarity (smartest)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    chunk_index: int


class BaseChunker(ABC):
    """Abstract chunker. All strategies produce List[Chunk]."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        ...

    @staticmethod
    def name() -> str:
        return "base"


# ============================================================
#  Strategy 1: Fixed-size character windows
# ============================================================


class FixedSizeChunker(BaseChunker):
    """Split by character count with overlap. Fast, predictable, no NLP.

    Best for: quick prototyping, documents with uniform structure.
    Weakness:  splits mid-sentence, can break semantic units.
    """

    def split(self, text: str) -> list[Chunk]:
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size // 2

        i = 0
        idx = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end].strip()
            if len(chunk_text) > 20:
                chunks.append(Chunk(
                    text=chunk_text, start_char=i, end_char=end, chunk_index=idx
                ))
                idx += 1
            i += step

        return chunks

    @staticmethod
    def name() -> str:
        return "fixed_size"


# ============================================================
#  Strategy 2: Sentence-boundary chunking
# ============================================================


class SentenceChunker(BaseChunker):
    """Split at sentence boundaries, grouping until chunk_size.

    Best for: QA over well-written prose (papers, articles, books).
    Weakness:  assumes clear sentence boundaries (not great for code/tables).

    Uses a regex that handles Chinese (。/！/？) and English (.!?) terminators.
    """

    _SENTENCE_RE = re.compile(r'(?<=[。！？.!?])\s+')

    def split(self, text: str) -> list[Chunk]:
        sentences = self._SENTENCE_RE.split(text)
        if len(sentences) <= 1:
            sentences = [text]

        chunks = []
        current = ""
        start = 0
        idx = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if len(current) + len(sent) <= self.chunk_size:
                current = (current + " " + sent).strip() if current else sent
            else:
                if len(current) > 20:
                    chunks.append(Chunk(
                        text=current,
                        start_char=start,
                        end_char=start + len(current),
                        chunk_index=idx,
                    ))
                    idx += 1
                # Overlap: carry last sentences forward
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    overlap_text = current[-self.chunk_overlap:]
                    current = overlap_text + " " + sent
                else:
                    current = sent
                start += len(chunks[-1].text) if chunks else 0

        if current and len(current) > 20:
            chunks.append(Chunk(
                text=current,
                start_char=start,
                end_char=start + len(current),
                chunk_index=idx,
            ))

        return chunks

    @staticmethod
    def name() -> str:
        return "sentence"


# ============================================================
#  Strategy 3: Semantic chunking (embedding-based topic boundary)
# ============================================================


class SemanticChunker(BaseChunker):
    """Split where embedding similarity drops — detecting topic shifts.

    How it works:
      1. Split text into sentences
      2. Compute embedding for each sentence (via ChromaDB's built-in model)
      3. Group consecutive sentences while cosine similarity > threshold
      4. When similarity drops below threshold → new chunk (topic shift!)

    Best for: diverse documents where topic boundaries matter.
    Weakness:  requires embedding model, slower, threshold tuning needed.

    Uses ChromaDB's built-in embedding function (all-MiniLM-L6-v2),
    same as what's used for later retrieval — no extra dependencies.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        similarity_threshold: float = 0.5,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self._ef = None

    def _get_embedding_fn(self):
        if self._ef is None:
            import chromadb.utils.embedding_functions as ef
            self._ef = ef.DefaultEmbeddingFunction()
        return self._ef

    def split(self, text: str) -> list[Chunk]:
        sentences = SentenceChunker._SENTENCE_RE.split(text)
        if len(sentences) <= 1:
            return SentenceChunker(
                self.chunk_size, self.chunk_overlap
            ).split(text)

        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return FixedSizeChunker(self.chunk_size, self.chunk_overlap).split(text)

        ef = self._get_embedding_fn()
        embeddings = ef(sentences)

        # Group sentences by embedding similarity
        groups = []
        current_group = [sentences[0]]
        current_emb = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = self._cosine_sim(current_emb, embeddings[i])
            current_len = sum(len(s) for s in current_group)

            if similarity >= self.similarity_threshold and current_len < self.chunk_size:
                current_group.append(sentences[i])
                # Update running average embedding
                current_emb = [
                    (current_emb[j] * (len(current_group) - 1) + embeddings[i][j])
                    / len(current_group)
                    for j in range(len(current_emb))
                ]
            else:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_emb = embeddings[i]

        if current_group:
            groups.append(current_group)

        # Build chunks from groups, respecting max size
        chunks = []
        idx = 0
        pos = 0
        for group in groups:
            chunk_text = " ".join(group)
            if len(chunk_text) > self.chunk_size * 2:
                # Group too large — fall back to fixed-size for this segment
                sub = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
                for c in sub.split(chunk_text):
                    c.chunk_index = idx
                    chunks.append(c)
                    idx += 1
            elif len(chunk_text) > 20:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=pos,
                    end_char=pos + len(chunk_text),
                    chunk_index=idx,
                ))
                idx += 1
            pos += len(chunk_text)

        return chunks

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def name() -> str:
        return "semantic"


# ============================================================
#  Factory
# ============================================================


def create_chunker(strategy: str, chunk_size: int = 800, chunk_overlap: int = 150) -> BaseChunker:
    """Factory: name → chunker instance."""
    mapping = {
        "fixed_size": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }
    cls = mapping.get(strategy, SentenceChunker)
    if strategy == "semantic":
        return cls(chunk_size, chunk_overlap)
    return cls(chunk_size, chunk_overlap)
