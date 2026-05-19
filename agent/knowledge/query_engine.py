"""RAG Query Engine — search papers + LLM synthesizes research-backed answers."""

import logging
import time

logger = logging.getLogger(__name__)


class KnowledgeQueryEngine:
    """RAG over AI research papers. Uses ChromaDB index + DeepSeek LLM."""

    def __init__(self, indexer=None, collector=None, llm_client=None, llm_model="deepseek-chat"):
        self.indexer = indexer
        self.collector = collector
        self.llm = llm_client
        self.llm_model = llm_model

    def search(self, query: str, top_k: int = 5) -> dict:
        results = []
        if self.indexer and self.indexer.is_built and self.indexer.collection:
            try:
                if self.indexer.embed_fn:
                    q_emb = self.indexer.embed_fn(query)
                    r = self.indexer.collection.query(query_embeddings=q_emb, n_results=top_k)
                else:
                    r = self.indexer.collection.query(query_texts=[query], n_results=top_k)

                if r and r.get("ids") and r["ids"][0]:
                    for i, pid in enumerate(r["ids"][0]):
                        meta = r.get("metadatas", [[{}]])[0][i] if r.get("metadatas") else {}
                        doc = r.get("documents", [[""]])[0][i] if r.get("documents") else ""
                        results.append({
                            "id": pid,
                            "title": meta.get("title", ""),
                            "authors": meta.get("authors", ""),
                            "abstract": doc[:500],
                            "relevance_score": 1.0,
                            "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
                        })
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        if not results and self.collector:
            for paper in self.collector.search_local(query)[:top_k]:
                results.append({
                    "id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors[:3]),
                    "abstract": paper.abstract[:500],
                    "relevance_score": 0.5,
                    "pdf_url": paper.pdf_url,
                })

        return {"results": results, "total": len(results)}

    async def query(self, question: str, top_k: int = 5, include_sources: bool = True) -> dict:
        start = time.time()
        sources = self.search(question, top_k=top_k).get("results", [])

        if not sources:
            return {"answer": "No relevant research found for this question.", "sources": [], "latency_seconds": 0}

        if not self.llm:
            return {
                "answer": "LLM not available. Here are the most relevant papers.",
                "sources": sources,
                "latency_seconds": round(time.time() - start, 2),
            }

        # Build context: prefer full text, fall back to abstract
        context_parts = []
        for i, s in enumerate(sources):
            paper_id = s.get("id", "")
            full_text = ""
            if paper_id and self.collector:
                full_text = self.collector.get_full_text(paper_id)
            body = full_text[:1500] if full_text else s.get("abstract", "")[:800]
            context_parts.append(
                f"[{i+1}] {s['title']}\nAuthors: {s.get('authors', '')}\n{body}"
            )
        context = "\n\n".join(context_parts)

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
            return {
                "answer": answer,
                "sources": sources if include_sources else [],
                "latency_seconds": round(time.time() - start, 2),
            }
        except Exception as e:
            return {
                "answer": f"Error synthesizing answer: {e}",
                "sources": sources,
                "latency_seconds": round(time.time() - start, 2),
            }

    def get_available(self) -> bool:
        return self.indexer is not None and self.indexer.is_built
