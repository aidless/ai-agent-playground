"""RAG Q&A System — Upload documents, ask questions, get cited answers.

Usage:
  uv run python -m rag_qa_system.main ingest <path>   # Load documents
  uv run python -m rag_qa_system.main ask <question>   # Ask a question
  uv run python -m rag_qa_system.main chat             # Interactive mode
  uv run python -m rag_qa_system.main stats            # Show collection stats
"""

import os
import sys
from pathlib import Path

try:
    from .ingest import ingest_directory, collection_stats
    from .query import ask, search_chunks
except ImportError:
    from rag_qa_system.ingest import ingest_directory, collection_stats
    from rag_qa_system.query import ask, search_chunks


def _print_answer(result):
    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  - {s}")
    print(f"\n({result.chunks_retrieved} chunks retrieved)")


def cmd_ingest(args: list[str]):
    if len(args) < 1:
        print("Usage: uv run python -m rag_qa_system.main ingest <path>")
        sys.exit(1)
    path = args[0]
    collection = args[1] if len(args) > 1 else "default"
    result = ingest_directory(path, collection)
    print(f"\nCollection: {result.collection_name}")
    print(f"Files: {result.files_processed}")
    print(f"Chunks: {result.chunks_created}")
    print(f"Ready for questions!")


def cmd_ask(args: list[str]):
    if len(args) < 1:
        print("Usage: uv run python -m rag_qa_system.main ask <question>")
        sys.exit(1)
    question = " ".join(args)
    collection = "default"
    result = ask(question, collection)
    _print_answer(result)


def cmd_chat():
    print("=" * 60)
    print("  RAG Q&A — Chat Mode")
    print("  Ask questions about your documents.")
    print("  Type 'sources' to see chunk details, 'quit' to exit.")
    print("=" * 60)
    print()

    collection = "default"
    stats = collection_stats(collection)
    if not stats:
        print("No documents ingested yet. Use 'ingest' command first.\n")

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
        if user_input.lower() == "sources":
            print("  Enter a search query to see relevant chunks:")
            q = input("  Search: ").strip()
            if q:
                chunks = search_chunks(q, collection)
                for c in chunks:
                    print(f"  [Chunk {c['chunk_index']}] {c['source']} "
                          f"(distance: {c['distance']:.3f})")
                    print(f"    {c['text']}\n")
            continue

        result = ask(user_input, collection)
        _print_answer(result)


def cmd_stats():
    stats = collection_stats("default")
    if stats:
        print(f"Collection: {stats['name']}")
        print(f"Chunks stored: {stats['chunks']}")
    else:
        print("No collection found. Ingest some documents first.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "ingest":
        cmd_ingest(args)
    elif cmd == "ask":
        cmd_ask(args)
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: ingest, ask, chat, stats")
        sys.exit(1)


if __name__ == "__main__":
    main()
