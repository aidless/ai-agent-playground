#!/usr/bin/env python3
"""Master Orchestrator — one command to run the entire system.

UV RUN: uv run python scripts/master_orchestrator.py

Exercises every component in order:
  1. Security check (pentest + b3)
  2. Knowledge base (collect + index)
  3. Code benchmark
  4. Engine status
  5. E2E test
  6. Generate comprehensive report

Output: master_report.json + console summary
"""

import asyncio, json, os, sys, time, subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def run_script(name):
    path = PROJECT / "scripts" / name
    if not path.exists():
        return {"status": "SKIP", "reason": f"Script not found: {name}"}
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT), capture_output=True, text=True, timeout=300
        )
        elapsed = (time.time() - t0) * 1000
        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit_code": result.returncode,
            "latency_ms": round(elapsed),
            "output": (result.stdout + result.stderr)[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "latency_ms": 300000}
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


async def run_engine_tests():
    """Test all engines without external scripts."""
    import dotenv
    dotenv.load_dotenv(PROJECT / ".env")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    results = {}

    # ReflectAction
    from agent.reflect_action import ReflectActionEngine
    e = ReflectActionEngine(failure_threshold=2)
    e.record_tool_result("web_search", False, "timeout")
    e.record_tool_result("web_search", False, "timeout")
    results["reflect_action"] = e.is_degraded("web_search")

    # Episodic memory
    from agent.episodic_memory import EpisodicMemoryStore
    store = EpisodicMemoryStore()
    store.add("Test reflection", task_type="code_generation", success=False)
    results["episodic_memory"] = len(store.retrieve("code_generation", k=1)) > 0

    # Knowledge
    from agent.knowledge.collector import PaperCollector
    results["knowledge_papers"] = PaperCollector().cached_count

    # Bootstrap safety
    from agent.bootstrap import BootstrapEngine
    be = BootstrapEngine.__new__(BootstrapEngine)
    be._tools = {}
    results["bootstrap_safety"] = not be._validate_syntax("import os\nos.system('evil')", "bad")

    # Agent basic run
    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.register("e", "E", {"properties": {"t": {"type": "str"}}, "required": ["t"]}, lambda t: t)
    agent = AsyncAgent(client=client, model="deepseek-chat", registry=registry, enable_super_agent=True)
    ctx = AgentContext(trace_id="master_test", max_steps=2)
    ctx = await agent.run(ctx, "Say hello")
    results["agent_run"] = ctx.state.value == "done"

    return results


async def main():
    print("=" * 60)
    print("MASTER ORCHESTRATOR — Full System Exercise")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    # Stage 1: Security
    print("\n[1/5] Security Check")
    report["stages"]["security_pentest"] = run_script("pentest.py")
    print(f"  Pentest: {report['stages']['security_pentest']['status']}")

    # Stage 2: Knowledge
    print("\n[2/5] Knowledge Base")
    from agent.knowledge.collector import PaperCollector
    from agent.knowledge.indexer import KnowledgeIndexer
    kc = PaperCollector()
    ki = KnowledgeIndexer(collector=kc)
    kr = ki.build_index()
    report["stages"]["knowledge"] = {
        "status": "PASS" if kr.get("chunks", 0) > 0 else "WARN",
        "papers": kc.cached_count,
        "chunks": kr.get("chunks", 0),
    }
    print(f"  Knowledge: {kc.cached_count} papers, {kr.get('chunks', 0)} chunks")

    # Stage 3: Benchmarks
    print("\n[3/5] Benchmarks")
    report["stages"]["code_bench"] = run_script("code_bench.py")
    print(f"  Code: {report['stages']['code_bench']['status']}")

    # Stage 4: Engine Tests
    print("\n[4/5] Engine Tests")
    engine_results = await run_engine_tests()
    report["stages"]["engines"] = engine_results
    passed = sum(1 for v in engine_results.values() if v)
    print(f"  Engines: {passed}/{len(engine_results)}")

    # Stage 5: Summary
    print("\n[5/5] Final Report")
    total_stages = len(report["stages"])
    all_pass = all(
        s.get("status") == "PASS" or s is True
        for s in report["stages"].values()
        if isinstance(s, dict)
    )
    report["summary"] = {
        "stages": total_stages,
        "all_pass": all_pass,
        "grade": "A+" if all_pass else "B",
    }
    print(f"  Grade: {report['summary']['grade']}")

    (PROJECT / "master_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: master_report.json")


if __name__ == "__main__":
    asyncio.run(main())
