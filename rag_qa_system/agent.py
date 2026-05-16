"""RAGAgent — Pipeline-style retrieval-augmented generation.

Pipeline: question → retrieve (ChromaDB) → generate (LLM) → cited answer
"""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import RAGConfig
from .ingest import DocumentIngester
from .query import RAGQuerier


class RAGAgent(BaseAgent):
    """Ask questions about your documents, get cited answers.

    Pipeline:
        preprocess:   question str → {"question": str, "context_chunks": [...]}
        _forward:     question + context → LLM → answer with citations
        postprocess:  raw answer → cleaned, formatted result
    """

    config_class = RAGConfig

    def __init__(self, config: RAGConfig | None = None):
        super().__init__(config)
        self.ingester = DocumentIngester(self.config)
        self.querier = RAGQuerier(self.config, self.llm)

    # ---- Pipeline implementation ----

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """Validate question, fetch relevant chunks from ChromaDB."""
        return {"question": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Retrieve + generate cited answer."""
        result = self.querier.ask(model_inputs["question"])
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,
            "chunks_retrieved": result.chunks_retrieved,
        }

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """Format the answer for display."""
        lines = [
            f"Q: {model_outputs['question']}",
            "",
            model_outputs["answer"],
            "",
        ]
        if model_outputs["sources"]:
            lines.append("Sources:")
            for s in model_outputs["sources"]:
                lines.append(f"  - {s}")
            lines.append(f"  ({model_outputs['chunks_retrieved']} chunks retrieved)")
        return "\n".join(lines)

    # ---- High-level API ----

    def ingest(self, path: str):
        """Load documents into the vector database."""
        return self.ingester.ingest(path)

    def ask(self, question: str) -> str:
        """Ask a question, get a cited answer."""
        return self.run(question)

    def chat(self):
        """Interactive Q&A session."""
        stats = self.ingester.stats()
        if not stats:
            print("No documents ingested. Use 'ingest <path>' first.\n")
            return

        print("=" * 60)
        print(f"  RAG Q&A — {stats['chunks']} chunks in '{stats['name']}'")
        print("  Type 'sources <query>' to debug, 'quit' to exit")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower().startswith("sources "):
                q = user_input[8:]
                chunks = self.querier.search(q)
                for c in chunks:
                    print(f"  [Chunk {c['chunk_index']}] {c['source']} "
                          f"(distance: {c['distance']:.3f})")
                    print(f"    {c['text']}\n")
                continue

            print(self.run(user_input))
            print()
