"""Quickstart — 5 minutes to first agent interaction.

Usage:
    uv run python examples/quickstart.py
"""

import asyncio
import os
import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()


async def main():
    # 1. Create client
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # 2. Setup agent
    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("echo", "Echo back input",
                     {"properties": {"text": {"type": "str"}}, "required": ["text"]},
                     lambda text: text)

    agent = AsyncAgent(
        client=client,
        model="deepseek-chat",
        registry=registry,
        enable_reflection=True,
        enable_learning=True,
    )

    # 3. Run a task
    task = "What is 2+2? Answer in one word."
    ctx = AgentContext(trace_id="quickstart_001")
    ctx = await agent.run(ctx, task)

    # 4. Print result
    for msg in ctx.messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            print(f"Agent: {msg['content']}")

    print(f"\nSteps: {ctx.step}")
    print(f"Reflections: {len(ctx.reflections)}")
    print(f"Lessons learned: {len(ctx.lessons)}")


if __name__ == "__main__":
    asyncio.run(main())
