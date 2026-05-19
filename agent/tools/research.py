"""Research Paper Tool — lets the agent query AI research papers.

Wraps the Knowledge Engine as a callable tool for the agent loop.
The agent can call 'research_paper' to search ArXiv papers and get
LLM-synthesized answers backed by research.

Usage in agent:
    result = agent.call_tool("research_paper",
        {"query": "What is the latest on ReAct agents?"})
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResearchPaperTool:
    """Agent-callable tool for querying AI research papers.

    Registered as 'research_paper' in the ToolRegistry.
    Calls the Knowledge Engine directly (same process — no HTTP overhead).
    """

    def __init__(self):
        self._engine = None   # Set by server.py after initialization

    def set_engine(self, engine):
        self._engine = engine

    def __call__(self, params: dict) -> str:
        """Execute research query. Params: query (str), top_k (int, optional)."""
        query = params.get("query", "")
        top_k = int(params.get("top_k", 5))

        if not query:
            return "Error: query parameter required"

        if not self._engine:
            return "Knowledge engine not initialized. Try again later."

        try:
            # Run async query synchronously (agent loop is async)
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._engine.query(query, context_size=top_k))
            answer = result.get("answer", "No answer")
            sources = result.get("sources", [])
            if sources:
                answer += "\n\nReferences:\n" + "\n".join(
                    f"  [{i+1}] {s.get('title','')}" for i, s in enumerate(sources[:5])
                )
            return answer
        except RuntimeError:
            # No event loop — create one
            result = asyncio.run(self._engine.query(query, context_size=top_k))
            answer = result.get("answer", "No answer")
            return answer
        except Exception as e:
            return f"Research query failed: {e}"


# Singleton
research_tool = ResearchPaperTool()
