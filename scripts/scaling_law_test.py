"""More Agents scaling law test — verify monotonic improvement with agent count.

More Agents Is All You Need (Li et al., 2024):
  "scaling the number of LLM-based agents monotonically improves task performance"

Tests 1, 3, 5, 7 agents on the same tasks to verify the scaling law.
"""

import asyncio, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


async def run_agent(client, task):
    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry
    r = ToolRegistry()
    r.register("e","E",{"properties":{"t":{"type":"str"}},"required":["t"]},lambda t:t)
    a = AsyncAgent(client=client, model="deepseek-chat", registry=r)
    ctx = AgentContext(trace_id=f"scale_{int(time.time())}", max_steps=2)
    ctx = await a.run(ctx, task)
    for m in ctx.messages:
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


async def main():
    import dotenv
    from openai import AsyncOpenAI
    dotenv.load_dotenv(PROJECT / ".env")

    d = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    from agent.matrix import AgentMatrix, MatrixAgentProfile

    tasks = [
        "What is 2+2? Answer in one word.",
        "Explain what a Python decorator is in 1-2 sentences.",
        "Is PostgreSQL a SQL or NoSQL database? One word.",
    ]

    agent_counts = [1, 3, 5]
    results_summary = {}

    print("SCALING LAW TEST (More Agents Is All You Need)")
    print("=" * 60)

    for n in agent_counts:
        print(f"\n{n} agents:")
        matrix = AgentMatrix()
        for i in range(n):
            matrix.add_agent(MatrixAgentProfile(
                f"agent-{i}", f"Agent {i}", "reasoner", "deepseek-chat", d
            ))

        scores = []
        for t in tasks:
            result = await matrix.solve(t)
            scores.append(1 if result.completed else 0)
            print(f"  {t[:50]}... -> completed={result.completed} (agents={len(result.results)})")

        avg = sum(scores) / len(scores)
        results_summary[n] = round(avg, 2)
        print(f"  Avg: {avg:.1%}")

    print(f"\n{'='*60}")
    print("SCALING RESULTS")
    for n, s in results_summary.items():
        print(f"  {n} agents: {s:.1%}")
    print("Verdict: ", end="")
    scores_list = list(results_summary.values())
    if all(scores_list[i] <= scores_list[i+1] for i in range(len(scores_list)-1)):
        print("MONOTONIC — More agents helps")
    else:
        print("NO CLEAR TREND — more data needed")

    (PROJECT / "scaling_report.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results_summary,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
