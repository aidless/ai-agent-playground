"""Knowledge Engine — ArXiv paper collector + vector index + research RAG.

Integrates into the Agent Playground so the agent can query the latest
AI research to improve its own decisions.

Components:
  1. PaperCollector — fetches from ArXiv API, caches metadata
  2. VectorIndex — builds ChromaDB index over papers
  3. KnowledgeQuery — RAG-based question answering over research
  4. Agent Tool — "research_paper" callable for the agent loop

Uses existing infrastructure: ChromaDB, sentence-transformers, DeepSeek.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "memory" / "knowledge"
PAPERS_DIR = KNOWLEDGE_DIR / "papers"
CHROMA_COLLECTION = "agent_research"

# Core search topics (focused, high-quality)
SEARCH_TOPICS = [
    "LLM agent autonomous system",
    "multi-agent debate collaboration",
    "AI agent self-improvement evolution",
    "agent tool learning function calling",
    "agent security safety alignment",
    "ReAct agent chain-of-thought reasoning",
    "retrieval augmented generation agent",
    "software engineering agent code repair",
    "agent evaluation benchmark measurement",
    "hypercent self-referential meta-learning",
]


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    categories: list[str]
    pdf_url: str

    @property
    def chunk_text(self) -> str:
        return f"Title: {self.title}\nAuthors: {', '.join(self.authors)}\nAbstract: {self.abstract}"


class PaperCollector:
    """Fetches AI agent research papers from ArXiv API.

    Usage:
        collector = PaperCollector()
        papers = await collector.fetch_topic("LLM agent", max_results=10)
    """

    def __init__(self):
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PaperMetadata] = {}
        self._load_cache()

    def _load_cache(self):
        cache_path = KNOWLEDGE_DIR / "paper_cache.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            for pid, pdata in data.items():
                self._cache[pid] = PaperMetadata(**pdata)

    def _save_cache(self):
        cache_path = KNOWLEDGE_DIR / "paper_cache.json"
        data = {pid: p.__dict__ for pid, p in self._cache.items()}
        cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    async def fetch_topic(self, topic: str, max_results: int = 10) -> list[PaperMetadata]:
        """Fetch papers for a topic from ArXiv API."""
        papers = []
        try:
            import urllib.request
            query = urllib.parse.quote(topic)
            url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{query}&start=0&max_results={max_results}"
                f"&sortBy=relevance&sortOrder=descending"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Agent-Playground/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")

            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                title = " ".join(entry.find("atom:title", ns).text.split())
                abstract = " ".join(entry.find("atom:summary", ns).text.split())

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)

                published = entry.find("atom:published", ns).text
                categories = [c.get("term") for c in entry.findall("atom:category", ns)]

                paper = PaperMetadata(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    categories=categories,
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                )

                if arxiv_id not in self._cache:
                    self._cache[arxiv_id] = paper
                    papers.append(paper)
                else:
                    papers.append(self._cache[arxiv_id])

            self._save_cache()
            logger.info("PaperCollector: fetched %d papers for '%s'", len(papers), topic)
        except Exception as e:
            logger.warning("PaperCollector fetch failed for '%s': %s", topic, e)

        return papers

    def get_paper(self, arxiv_id: str) -> Optional[PaperMetadata]:
        return self._cache.get(arxiv_id)

    def search_local(self, keyword: str) -> list[PaperMetadata]:
        """Search cached papers by keyword."""
        results = []
        k = keyword.lower()
        for paper in self._cache.values():
            score = 0
            if k in paper.title.lower():
                score += 3
            if k in paper.abstract.lower():
                score += 1
            for author in paper.authors:
                if k in author.lower():
                    score += 2
            if score > 0:
                results.append((score, paper))
        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results[:20]]

    @property
    def cached_count(self) -> int:
        return len(self._cache)


class KnowledgeEngine:
    """RAG engine over AI research papers, integrated into the agent loop.

    Usage:
        engine = KnowledgeEngine()
        papers = await engine.collect_papers()
        engine.build_index()

        # Agent calls:
        result = await engine.query("What is the latest on ReAct agents?")
        print(result)
    """

    def __init__(self, llm_client=None, llm_model: str = "deepseek-chat"):
        self.llm = llm_client
        self.llm_model = llm_model
        self.collector = PaperCollector()
        self._index_built = False
        self._chroma_client = None
        self._collection = None
        self._embed_fn = None

    async def collect_papers(self, max_per_topic: int = 5) -> int:
        """Fetch papers across all AI agent topics."""
        total = 0
        for topic in SEARCH_TOPICS:
            papers = await self.collector.fetch_topic(topic, max_results=max_per_topic)
            total += len(papers)
            time.sleep(0.5)  # Rate limit
        logger.info("KnowledgeEngine: collected %d total papers", total)
        return total

    def build_index(self):
        """Build ChromaDB vector index over collected papers."""
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(KNOWLEDGE_DIR / "chroma"))

            # Delete and rebuild if exists
            try:
                self._chroma_client.delete_collection(CHROMA_COLLECTION)
            except Exception:
                pass

            self._collection = self._chroma_client.create_collection(
                name=CHROMA_COLLECTION,
                metadata={"description": "AI Agent research papers"},
            )

            # Use sentence-transformers for embeddings
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_fn = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not available, using simple TF-IDF")
                self._embed_fn = None

            papers = list(self.collector._cache.values())
            if not papers:
                logger.warning("KnowledgeEngine: no papers to index")
                return

            batch_size = 20
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i + batch_size]
                ids = [p.arxiv_id for p in batch]
                texts = [p.chunk_text for p in batch]
                metadatas = [{"title": p.title, "authors": ", ".join(p.authors[:3]), "published": p.published} for p in batch]

                if self._embed_fn:
                    embeddings = self._embed_fn.encode(texts).tolist()
                    self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
                else:
                    self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

            self._index_built = True
            logger.info("KnowledgeEngine: indexed %d papers", len(papers))
        except ImportError:
            logger.warning("KnowledgeEngine: chromadb not available — index not built")
        except Exception as e:
            logger.warning("KnowledgeEngine: index build failed: %s", e)

    def _simple_search(self, query: str, k: int = 5) -> list[dict]:
        """Fallback keyword search over cached papers."""
        results = self.collector.search_local(query)
        return [
            {"id": p.arxiv_id, "title": p.title, "abstract": p.abstract[:500],
             "authors": p.authors, "url": p.pdf_url}
            for p in results[:k]
        ]

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for relevant papers using vector similarity or keyword."""
        if self._index_built and self._collection:
            try:
                if self._embed_fn:
                    q_embedding = self._embed_fn.encode([query]).tolist()
                    results = self._collection.query(query_embeddings=q_embedding, n_results=k)
                else:
                    results = self._collection.query(query_texts=[query], n_results=k)

                papers = []
                if results and results.get("ids") and results["ids"][0]:
                    for i, pid in enumerate(results["ids"][0]):
                        meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
                        doc = results.get("documents", [[""]])[0][i] if results.get("documents") else ""
                        papers.append({
                            "id": pid,
                            "title": meta.get("title", ""),
                            "authors": meta.get("authors", ""),
                            "abstract": doc[:500],
                        })
                return papers
            except Exception as e:
                logger.warning("KnowledgeEngine vector search failed: %s", e)

        return self._simple_search(query, k)

    async def query(self, question: str, context_size: int = 5) -> dict:
        """RAG query: search papers → LLM synthesizes answer."""
        papers = self.search(question, k=context_size)

        if not papers:
            return {"answer": "No relevant research found for this question.", "sources": []}

        if not self.llm:
            return {
                "answer": "LLM not available. Here are the most relevant papers.",
                "sources": papers,
            }

        # Build context from papers
        context = "\n\n".join(
            f"[{i+1}] {p['title']}\nAuthors: {p.get('authors', '')}\nAbstract: {p.get('abstract', '')[:800]}"
            for i, p in enumerate(papers)
        )

        prompt = (
            f"Based on the following research papers, answer this question about AI Agents:\n\n"
            f"Question: {question}\n\n"
            f"Research papers:\n{context}\n\n"
            f"Provide a clear, well-cited answer. Reference paper numbers [1], [2], etc.\n"
            f"If the papers don't cover the question, say so honestly."
        )

        try:
            response = await self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            return {"answer": answer, "sources": papers}
        except Exception as e:
            return {"answer": f"Error synthesizing answer: {e}", "sources": papers}

    def get_agent_tool(self):
        """Return a callable tool for the agent to use.

        Usage:
            tool = engine.get_agent_tool()
            result = tool("What is the state of the art in agent evaluation?")
        """
        async def research_paper(question: str) -> str:
            result = await self.query(question)
            answer = result.get("answer", "No answer")
            sources = result.get("sources", [])
            if sources:
                answer += "\n\nSources:\n" + "\n".join(
                    f"[{i+1}] {s.get('title', '')} ({s.get('id', '')})"
                    for i, s in enumerate(sources[:5])
                )
            return answer

        return research_paper

    def status(self) -> dict:
        return {
            "papers_cached": self.collector.cached_count,
            "index_built": self._index_built,
            "collection": CHROMA_COLLECTION,
            "topics": len(SEARCH_TOPICS),
        }
