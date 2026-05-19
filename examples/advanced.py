"""Advanced — SuperAgent with debate, evolution, and self-play.

Usage:
    uv run python examples/advanced.py
"""

import asyncio, os, dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()


async def main():
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry
    from agent.debate import DebateEngine

    # Setup with all SuperAgent features
    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda t: t)

    agent = AsyncAgent(
        client=client,
        model="deepseek-chat",
        registry=registry,
        enable_super_agent=True,
    )

    # 1. Debate mode
    debate = DebateEngine(client, client, client)
    result = await debate.debate(
        task="Explain the CAP theorem in one paragraph",
        primary_model="deepseek-chat",
        challenger_model="deepseek-chat",
    )
    print(f"Debate: {result.total_rounds} rounds")
    print(f"Consensus: {result.consensus[:200]}...")

    # 2. Self-play training
    from agent.self_play import SelfPlayEngine
    sp = SelfPlayEngine(agent, client)
    results = await sp.train(rounds=3)
    print(f"\nSelf-play: {len(results)} rounds completed")
    for r in results:
        print(f"  {r.task.domain}: {r.score}/10")

    # 3. Autonomous agent loop
    task = "Explain what makes a good REST API design. Be concise."
    ctx = AgentContext(trace_id="advanced_001", max_steps=2)
    ctx = await agent.run(ctx, task)
    for msg in ctx.messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            print(f"\nAgent: {msg['content'][:300]}...")

    # 4. Show super status
    print(f"\nEngine status: {agent.get_super_status().keys()}")


if __name__ == "__main__":
    asyncio.run(main())
