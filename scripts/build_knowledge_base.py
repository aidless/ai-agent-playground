#!/usr/bin/env python3
"""Knowledge Base Builder — standalone script to batch-collect papers and build index.

Usage:
    uv run python scripts/build_knowledge_base.py          # Default: 50 papers
    uv run python scripts/build_knowledge_base.py 100      # Collect 100 papers
    uv run python scripts/build_knowledge_base.py --rebuild # Rebuild index from cache

Features:
  - Batch collect from ArXiv with retry + exponential backoff
  - Build ChromaDB vector index
  - Progress reporting + resume from cache
  - Works independently of running server
"""

import asyncio
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


async def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
    rebuild_only = "--rebuild" in sys.argv

    print("=" * 60)
    print("KNOWLEDGE BASE BUILDER")
    print(f"Target: {target} papers" if not rebuild_only else "Rebuild mode: index from cache")
    print("=" * 60)

    from agent.knowledge.collector import PaperCollector
    from agent.knowledge.indexer import KnowledgeIndexer

    collector = PaperCollector()

    if not rebuild_only:
        print(f"\nPre-collection cache: {collector.cached_count} papers")
        print(f"Collecting up to {target} papers...\n")

        result = collector.collect(max_papers=target)
        print(f"\nResult: collected {result['collected']} new papers")
        print(f"Total cached: {result['total']}")
        if result["papers"]:
            print("New:")
            for title in result["papers"][:10]:
                print(f"  {title[:80]}")

    print(f"\nBuilding vector index...")
    indexer = KnowledgeIndexer(collector=collector)
    build_result = indexer.build_index(force_rebuild=rebuild_only)

    if "error" in build_result:
        print(f"ERROR: {build_result['error']}")
    else:
        print(f"Indexed: {build_result.get('documents', 0)} documents")
        print(f"Chunks: {build_result.get('chunks', 0)}")

    status = indexer.get_status()
    print(f"\nFinal status:")
    print(f"  Papers: {collector.cached_count}")
    print(f"  Index chunks: {status['total_chunks']}")
    print(f"  Index size: {status['index_size_mb']:.1f} MB")
    print(f"  Embedding: {status['embedding_model']}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
