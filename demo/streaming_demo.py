"""
Streaming + concurrency demo — Agent output token by token.

Demonstrates:
  1. Streaming: agent outputs tokens as they're generated (no 30s white screen)
  2. Concurrent: two agents run simultaneously via asyncio

Usage:
  uv run python -m demo.streaming_demo
"""

import asyncio
import sys
from ai_agent_playground.base import ToolCallEvent


# ============================================================
#  Demo 1: Streaming agent
# ============================================================


def demo_streaming():
    """Stream a single agent question token by token."""
    from mcp_agent.agent import MCPToolAgent

    print("=" * 60)
    print("  Demo 1: Streaming Agent (token-by-token)")
    print("=" * 60)
    print()

    agent = MCPToolAgent()

    question = "Calculate sqrt(144) + 3^3, then write the result to streaming_result.txt"
    print(f"Q: {question}\n")
    print("A: ", end="", flush=True)

    try:
        for item in agent.run_stream(question):
            if isinstance(item, str):
                print(item, end="", flush=True)
            elif isinstance(item, ToolCallEvent):
                if item.phase == "start":
                    args_str = ", ".join(
                        f"{k}={v}" for k, v in (item.args or {}).items()
                    )
                    print(f"\n  ⚙ {item.tool_name}({args_str}) ...", end=" ", flush=True)
                else:
                    preview = (item.result or "")[:80].replace("\n", " ")
                    print(f"→ {preview}", flush=True)
                    print("  ", end="", flush=True)
    finally:
        agent.close()

    print("\n")


# ============================================================
#  Demo 2: Concurrent agents
# ============================================================


async def _ask_agent(agent, question: str, label: str):
    """Helper: ask an agent a question asynchronously."""
    result = await agent.arun(question)
    return label, result


async def demo_concurrent():
    """Run two agents simultaneously."""
    from mcp_agent.agent import MCPToolAgent

    print("=" * 60)
    print("  Demo 2: Concurrent Agents (2 agents in parallel)")
    print("=" * 60)
    print()

    q1 = "What is 25 * 4 + 10? Use the calculator."
    q2 = "What is 100 / 4 - 5? Use the calculator."

    agent1 = MCPToolAgent()
    agent2 = MCPToolAgent()

    print(f"Agent A: {q1}")
    print(f"Agent B: {q2}")
    print()
    print("Running both simultaneously...")
    print()

    start = asyncio.get_event_loop().time()

    try:
        results = await asyncio.gather(
            _ask_agent(agent1, q1, "A"),
            _ask_agent(agent2, q2, "B"),
        )
    finally:
        agent1.close()
        agent2.close()

    elapsed = asyncio.get_event_loop().time() - start

    for label, answer in results:
        # Extract just the key result line
        lines = [l for l in answer.split("\n") if l.strip()]
        preview = lines[0] if lines else answer
        print(f"Agent {label}: {preview[:150]}")

    print()
    print(f"Both completed in {elapsed:.1f}s (sequentially would be ~{elapsed*2:.1f}s)")
    print()


# ============================================================
#  Main
# ============================================================


def main():
    demo_streaming()

    if "--skip-concurrent" not in sys.argv:
        asyncio.run(demo_concurrent())

    print("=" * 60)
    print("  Done. Streaming + concurrent agents verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
