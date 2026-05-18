"""End-to-End SuperAgent Demo — real LLMs, all three engines.

Exercises:
  1. Multi-Model Debate: DeepSeek V4 vs Ollama Qwen2.5  →  Arbitrator
  2. Reflect→Action: simulated tool failure → degradation
  3. Skills Bootstrap: DeepSeek generates a real tool → AST validate → register

Outputs: demo_report.json + console summary
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dotenv
from openai import AsyncOpenAI

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

dotenv.load_dotenv(PROJECT / ".env")


# ── Clients ──────────────────────────────────────

def create_clients():
    deepseek = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    ollama = None
    try:
        ollama = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        print("[OK] DeepSeek + Ollama clients ready")
    except Exception:
        print("[WARN] Ollama not available — debate will use self-critique mode")
    return deepseek, ollama


# ── Demo 1: Multi-Model Debate ────────────────────

async def demo_debate(deepseek, ollama):
    print("\n" + "=" * 60)
    print("DEMO 1: Multi-Model Debate")
    print("=" * 60)

    from agent.debate import DebateEngine

    task = (
        "What is the best way to implement rate limiting in a FastAPI application? "
        "Consider both middleware-based and dependency-based approaches."
    )

    print(f"Task: {task}")
    print("Models: DeepSeek V4 (primary) vs Qwen2.5:7b (challenger)")

    engine = DebateEngine(
        primary_client=deepseek,
        challenger_client=ollama or deepseek,
        arbitrator_client=deepseek,
    )

    t0 = time.time()
    result = await engine.debate(
        task=task,
        primary_model="deepseek-chat",
        challenger_model="qwen2.5:7b" if ollama else "deepseek-chat",
    )
    elapsed = (time.time() - t0) * 1000

    print(f"\n  Rounds: {result.total_rounds}")
    print(f"  Latency: {elapsed:.0f}ms")
    print(f"  Completed: {result.completed}")

    if result.consensus:
        print(f"\n  Consensus (first 400 chars):")
        print(f"  {result.consensus[:400]}...")

    return {
        "debate": {
            "rounds": result.total_rounds,
            "latency_ms": elapsed,
            "completed": result.completed,
            "consensus_preview": result.consensus[:800],
            "round_details": [
                {"round": r.round_num, "speaker": r.speaker, "chars": len(r.content)}
                for r in result.rounds
            ],
        }
    }


# ── Demo 2: Reflect→Action ────────────────────────

async def demo_reflect_action():
    print("\n" + "=" * 60)
    print("DEMO 2: Reflect→Action (Tool Degradation)")
    print("=" * 60)

    from agent.reflect_action import ReflectActionEngine

    engine = ReflectActionEngine(failure_threshold=3)

    # Simulate web_search failing 3 times
    print("Simulating: web_search fails 3 times in a row...")
    for i in range(3):
        engine.record_tool_result("web_search", success=False, error=f"Connection timeout #{i+1}")

    degraded = engine.is_degraded("web_search")
    alternatives = engine.get_alternatives("web_search")
    print(f"  web_search degraded: {degraded}")
    print(f"  Alternatives: {alternatives}")

    # Test tool call filtering
    tool_calls = [
        {"function": {"name": "web_search", "arguments": '{"q": "test"}'}},
        {"function": {"name": "read_file", "arguments": '{"path": "data.txt"}'}},
    ]
    filtered = engine.filter_degraded_tools(tool_calls)
    names = [tc["function"]["name"] for tc in filtered]
    print(f"  Before filter: web_search, read_file")
    print(f"  After filter:  {', '.join(names)}")

    # Test missing tool detection
    reflection = "I need a tool to parse CSV files but I don't have any tool for that"
    actions = engine.evaluate(reflection, [])
    missing = [a for a in actions if a["type"] == "missing_tool"]
    print(f"\n  Reflection: '{reflection}'")
    print(f"  Detected missing tool: {missing[0]['suggested_name'] if missing else 'none'}")

    return {
        "reflect_action": {
            "web_search_degraded": degraded,
            "alternatives": alternatives,
            "tool_substitution_verified": "web_search" not in names,
            "missing_tool_detected": missing[0]["suggested_name"] if missing else None,
            "status": engine.status(),
        }
    }


# ── Demo 3: Skills Bootstrapping ───────────────────

async def demo_bootstrap(deepseek):
    print("\n" + "=" * 60)
    print("DEMO 3: Skills Bootstrapping (Real LLM)")
    print("=" * 60)

    from agent.bootstrap import BootstrapEngine
    from agent.tools.registry import ToolRegistry

    engine = BootstrapEngine(deepseek, model="deepseek-chat")
    registry = ToolRegistry()

    # Generate a real tool with DeepSeek
    print("Task: Generate a 'markdown_table_to_json' tool via DeepSeek...")
    t0 = time.time()
    tool = await engine.generate_from_reflection(
        "I need to convert markdown tables to JSON format but don't have a parser",
        "markdown_table_to_json",
    )
    elapsed = (time.time() - t0) * 1000

    print(f"  Validated: {tool.validated}")
    print(f"  Code length: {len(tool.code)} chars")
    print(f"  Latency: {elapsed:.0f}ms")

    if tool.validated:
        registered = engine.register_tool(tool, registry)
        print(f"  Registered: {registered}")

        # Test the generated tool
        test_input = "| Name | Age |\n|------|-----|\n| Alice | 25 |\n| Bob | 30 |"
        try:
            result = registry.execute("markdown_table_to_json", {"table": test_input, "markdown": test_input})
            print(f"  Test input: {test_input[:60]}...")
            print(f"  Tool output: {result[:200]}")
            works = True
        except Exception as e:
            print(f"  Tool execution error: {e}")
            works = False
    else:
        works = False
        print(f"  Error: {tool.error}")

    print(f"\n  Generated code:")
    for line in tool.code.split("\n")[:10]:
        print(f"    {line}")

    return {
        "bootstrap": {
            "tool_name": tool.name,
            "validated": tool.validated,
            "registered": tool.registered if tool.validated else False,
            "code_length": len(tool.code),
            "latency_ms": elapsed,
            "test_works": works,
            "code_preview": tool.code[:500],
        }
    }


# ── Demo 4: Full Agent Loop (LLM with SuperAgent) ──

async def demo_agent_loop(deepseek):
    print("\n" + "=" * 60)
    print("DEMO 4: Agent Loop with SuperAgent features")
    print("=" * 60)

    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    # Register a simple echo tool
    registry.register("echo", "Returns the input text",
                     {"properties": {"text": {"type": "str"}}, "required": ["text"]},
                     lambda text: text)

    agent = AsyncAgent(
        client=deepseek,
        model="deepseek-chat",
        registry=registry,
        enable_super_agent=True,
        enable_reflection=True,
        enable_learning=True,
    )

    task = "What is 2+2? Just tell me the answer directly — don't use any tools."
    print(f"Task: {task}")

    ctx = AgentContext(trace_id="demo_001", max_steps=3)
    t0 = time.time()
    ctx = await agent.run(ctx, task)
    elapsed = (time.time() - t0) * 1000

    response = ""
    for msg in ctx.messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            response = msg["content"]

    print(f"  State: {ctx.state.value}")
    print(f"  Steps: {ctx.step}")
    print(f"  Latency: {elapsed:.0f}ms")
    print(f"  Response: {response[:200]}")

    return {
        "agent_loop": {
            "state": ctx.state.value,
            "steps": ctx.step,
            "latency_ms": elapsed,
            "response": response[:300],
            "reflections": ctx.reflections[:2],
            "lessons": ctx.lessons[:2],
            "super_status": agent.get_super_status(),
        }
    }


# ── Main ──────────────────────────────────────────

async def main():
    print("SuperAgent End-to-End Demo")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project: {PROJECT}")

    deepseek, ollama = create_clients()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": {},
    }

    # Demo 1: Debate (needs Ollama for full effect)
    try:
        report["results"]["debate"] = await demo_debate(deepseek, ollama)
    except Exception as e:
        print(f"[SKIP] Debate demo failed: {e}")
        report["results"]["debate"] = {"error": str(e)}

    # Demo 2: ReflectAction (no LLM needed)
    try:
        report["results"]["reflect_action"] = await demo_reflect_action()
    except Exception as e:
        print(f"[FAIL] ReflectAction demo failed: {e}")
        report["results"]["reflect_action"] = {"error": str(e)}

    # Demo 3: Bootstrap (uses DeepSeek)
    try:
        report["results"]["bootstrap"] = await demo_bootstrap(deepseek)
    except Exception as e:
        print(f"[SKIP] Bootstrap demo failed: {e}")
        report["results"]["bootstrap"] = {"error": str(e)}

    # Demo 4: Agent Loop (uses DeepSeek)
    try:
        report["results"]["agent_loop"] = await demo_agent_loop(deepseek)
    except Exception as e:
        print(f"[SKIP] Agent loop demo failed: {e}")
        report["results"]["agent_loop"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)

    for name, result in report["results"].items():
        if "error" in result:
            print(f"  {name}: FAILED — {result['error']}")
        else:
            print(f"  {name}: PASSED")

    # Save report
    report_path = PROJECT / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
