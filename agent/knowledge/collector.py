"""Paper Collector — fetches AI agent research from ArXiv API."""

import json
import time
import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "memory" / "knowledge" / "papers"

DEFAULT_TOPICS = [
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


class PaperMetadata:
    def __init__(self, arxiv_id, title, abstract, authors, published, categories, pdf_url):
        self.arxiv_id = arxiv_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.published = published
        self.categories = categories
        self.pdf_url = pdf_url

    @property
    def chunk_text(self):
        return f"Title: {self.title}\nAuthors: {', '.join(self.authors)}\nAbstract: {self.abstract}"


class PaperCollector:
    """Fetches AI agent research papers from ArXiv API."""

    def __init__(self, data_dir: str = ""):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self._cache: dict[str, PaperMetadata] = {}
        self._load_cache()

    def _load_cache(self):
        if self.metadata_file.exists():
            data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            for pid, pdata in data.items():
                try:
                    self._cache[pid] = PaperMetadata(**pdata)
                except Exception:
                    pass
        # Load default papers if cache is empty
        if not self._cache:
            self._load_default_papers()

    def _load_default_papers(self):
        """Load bundled default papers when ArXiv is unreachable (China GFW)."""
        default_file = self.data_dir / "default_papers.json"
        if not default_file.exists():
            return
        try:
            data = json.loads(default_file.read_text(encoding="utf-8"))
            for p in data.get("papers", []):
                pid = p.get("id", "")
                if pid and pid not in self._cache:
                    try:
                        self._cache[pid] = PaperMetadata(
                            arxiv_id=pid,
                            title=p.get("title", ""),
                            abstract=p.get("abstract", ""),
                            authors=p.get("authors", []),
                            published=p.get("published", ""),
                            categories=p.get("categories", []),
                            pdf_url=f"https://arxiv.org/pdf/{pid}.pdf",
                        )
                    except Exception:
                        pass
            self._save_cache()
            logger.info("Loaded %d default papers (ArXiv offline fallback)", len(data.get("papers", [])))
        except Exception as e:
            logger.warning("Failed to load default papers: %s", e)

    def _save_cache(self):
        data = {pid: p.__dict__ for pid, p in self._cache.items()}
        self.metadata_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def collect(self, max_papers: int = 50, topics: Optional[list[str]] = None) -> dict:
        topics = topics or DEFAULT_TOPICS
        imported_ids = set()
        if self.metadata_file.exists():
            imported_ids = set(self._cache.keys())

        new_papers = []
        new_ids = []

        for topic in topics:
            if len(new_papers) >= max_papers:
                break

            logger.info("Collector: searching '%s'...", topic)
            papers = self._fetch_topic(topic, min(10, max_papers - len(new_papers)))

            for paper in papers:
                if paper.arxiv_id not in imported_ids:
                    self._cache[paper.arxiv_id] = paper
                    imported_ids.add(paper.arxiv_id)
                    new_papers.append(paper)
                    new_ids.append(paper.arxiv_id)
                    logger.info("  ✅ %s", paper.title[:60])

            time.sleep(1)

        self._save_cache()
        return {
            "collected": len(new_papers),
            "total": len(self._cache),
            "papers": [p.title for p in new_papers[:5]],
        }

    def _fetch_topic(self, topic: str, max_results: int = 10) -> list[PaperMetadata]:
        """Fetch papers for a topic with retry + exponential backoff."""
        import urllib.request
        import urllib.parse
        import urllib.error

        query = urllib.parse.quote(topic)
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{query}&start=0&max_results={max_results}"
            f"&sortBy=relevance&sortOrder=descending"
        )

        retries = 3
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AI-Agent-Playground/1.0"})
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = resp.read().decode("utf-8")

                root = ET.fromstring(data)
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
                papers = []
                for entry in root.findall("atom:entry", ns):
                    arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                    title = " ".join(entry.find("atom:title", ns).text.split())
                    abstract = " ".join(entry.find("atom:summary", ns).text.split())
                    authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)
                              if a.find("atom:name", ns) is not None]
                    published = entry.find("atom:published", ns).text
                    categories = [c.get("term") for c in entry.findall("atom:category", ns)]
                    papers.append(PaperMetadata(
                        arxiv_id, title, abstract, authors, published, categories,
                        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    ))
                return papers

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = (2 ** attempt) * 5
                    logger.warning("ArXiv 429 rate limit on '%s' — retry %d/%d after %ds",
                                 topic, attempt + 1, retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("ArXiv HTTP %d for '%s': %s", e.code, topic, e)
                    time.sleep(3)
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning("ArXiv timeout on '%s' — retry %d/%d", topic, attempt + 1, retries)
                    time.sleep(5)
                else:
                    logger.warning("ArXiv fetch failed for '%s' after %d retries: %s", topic, retries, e)

        return []

    def search_local(self, keyword: str) -> list[PaperMetadata]:
        k = keyword.lower()
        results = []
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

    def get_status(self) -> dict:
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return {
            "total_papers": len(self._cache),
            "metadata_count": len(self._cache),
            "storage_size_mb": sum(f.stat().st_size for f in pdf_files) / (1024 * 1024),
        }

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    def download_pdf(self, arxiv_id: str) -> str:
        """Download PDF and extract full text. Returns extracted text or empty string."""
        pdf_path = self.data_dir / f"{arxiv_id}.pdf"
        txt_path = self.data_dir / f"{arxiv_id}.txt"

        # Return cached text if available
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="replace")

        # Download PDF
        if not pdf_path.exists():
            try:
                url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                req = urllib.request.Request(url, headers={"User-Agent": "AI-Agent-Playground/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    pdf_path.write_bytes(resp.read())
                logger.info("Downloaded PDF: %s", arxiv_id)
            except Exception as e:
                logger.warning("PDF download failed for %s: %s", arxiv_id, e)
                return ""

        # Extract text from PDF
        try:
            text = ""
            # Try pypdf first
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            except ImportError:
                pass

            # Fallback: PyPDF2
            if not text:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(str(pdf_path))
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
                except ImportError:
                    pass

            if text:
                txt_path.write_text(text, encoding="utf-8", errors="replace")
                logger.info("Extracted full text: %s (%d chars)", arxiv_id, len(text))
                return text
        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", arxiv_id, e)

        return ""

    def get_full_text(self, arxiv_id: str) -> str:
        """Get full paper text, downloading PDF if needed."""
        return self.download_pdf(arxiv_id)
