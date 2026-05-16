"""RAG Q&A System — Upload documents, ask questions, get cited answers.

Usage:
  uv run python -m rag_qa_system.main ingest <path>   # Load documents
  uv run python -m rag_qa_system.main ask <question>   # Ask a question
  uv run python -m rag_qa_system.main chat             # Interactive mode
  uv run python -m rag_qa_system.main stats            # Show collection stats
"""

import sys
from pathlib import Path

from .agent import RAGAgent


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    agent = RAGAgent()

    if cmd == "ingest":
        if len(args) < 1:
            print("Usage: uv run python -m rag_qa_system.main ingest <path>")
            sys.exit(1)
        result = agent.ingest(args[0])
        print(f"\nCollection: {result.collection_name}")
        print(f"Files: {result.files_processed}")
        print(f"Chunks: {result.chunks_created}")
        print("Ready for questions!")

    elif cmd == "ask":
        if len(args) < 1:
            print("Usage: uv run python -m rag_qa_system.main ask <question>")
            sys.exit(1)
        question = " ".join(args)
        print(agent.ask(question))

    elif cmd == "chat":
        agent.chat()

    elif cmd == "stats":
        stats = agent.ingester.stats()
        if stats:
            print(f"Collection: {stats['name']}")
            print(f"Chunks stored: {stats['chunks']}")
        else:
            print("No collection found. Ingest some documents first.")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: ingest, ask, chat, stats")
        sys.exit(1)


if __name__ == "__main__":
    main()
