"""Engine Benchmark — measure how much each SuperAgent engine improves output quality.

Tests 5 questions across different domains. For each question:
  1. Baseline: single DeepSeek answer
  2. Debate: process-centric multi-model debate
  3. Matrix: multi-agent specialized routing

Scores all outputs with 3D Evaluation Gate. Generates comparison report.

Usage: uv run python scripts/benchmark_engines.py
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

BENCHMARK_QUESTIONS = [
    {
        "id": "q1_code",
        "domain": "coding",
        "task": "Write a Python function that checks if a string is a valid IPv4 address. Include test cases.",
    },
    {
        "id": "q2_reason",
        "domain": "reasoning",
        "task": "Explain why database indexes speed up queries but slow down writes. Be specific.",
    },
    {
        "id": "q3_security",
        "domain": "security",
        "task": "Explain three common authentication vulnerabilities in web apps and how to prevent each.",
    },
    {
        "id": "q4_design",
        "domain": "system_design",
        "task": "Design a URL shortener service. Describe the API, database schema, and scaling strategy.",
    },
    {
        "id": "q5_algo",
        "domain": "algorithms",
        "task": "Explain the difference between quicksort and mergesort. When would you use each?",
    },
]


async def run_benchmark():
    deepseek = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    ollama = None
    try:
        ollama = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        print("[OK] Ollama connected")
    except Exception:
        print("[WARN] Ollama not available — debate will use self-critique")

    from agent.eval_gate import EvaluationGate
    from agent.debate import DebateEngine
    from agent.matrix import AgentMatrix, MatrixAgentProfile
    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry

    gate = EvaluationGate(deepseek)
    debate_eng = DebateEngine(deepseek, ollama or deepseek, deepseek)

    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda text: text)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry, enable_super_agent=True)

    matrix = AgentMatrix()
    matrix.add_agent(MatrixAgentProfile("ds-reasoner", "DeepSeek Reasoner", "reasoner", "deepseek-chat", deepseek))
    matrix.add_agent(MatrixAgentProfile("ds-coder", "DeepSeek Coder", "coder", "deepseek-chat", deepseek))
    if ollama:
        matrix.add_agent(MatrixAgentProfile("qwen-reviewer", "Qwen Reviewer", "reviewer", "qwen2.5:7b", ollama))

    results = []
    print(f"\n{'='*70}")
    print(f"ENGINE BENCHMARK — {len(BENCHMARK_QUESTIONS)} questions, 3 engines each")
    print(f"{'='*70}")

    for q in BENCHMARK_QUESTIONS:
        print(f"\n--- {q['id']} ({q['domain']}): {q['task'][:60]}...")

        entry = {"question": q, "baseline": None, "debate": None, "matrix": None}
        task = q["task"]

        # ── Baseline ──
        t0 = time.time()
        ctx = AgentContext(trace_id=f"bench_bl_{q['id']}", max_steps=2)
        ctx = await agent.run(ctx, task)
        baseline = ""
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                baseline = msg["content"]
        bl_time = (time.time() - t0) * 1000
        bl_score = await gate.evaluate("baseline", candidate_text=baseline, task=task)
        entry["baseline"] = {
            "latency_ms": round(bl_time),
            "chars": len(baseline),
            "overall": bl_score.dimensions.overall if bl_score.dimensions else 5.0,
            "dimensions": {
                "interface": bl_score.dimensions.interface_score if bl_score.dimensions else 0,
                "functional": bl_score.dimensions.functional_score if bl_score.dimensions else 0,
                "utility": bl_score.dimensions.utility_score if bl_score.dimensions else 0,
            },
        }
        print(f"  Baseline: {bl_score.dimensions.overall if bl_score.dimensions else 5.0}/10 ({bl_time:.0f}ms)")

        # ── Debate ──
        try:
            t0 = time.time()
            debate_result = await debate_eng.debate_process_centric(task, "deepseek-chat", "qwen2.5:7b" if ollama else "deepseek-chat")
            debate_time = (time.time() - t0) * 1000
            db_score = await gate.evaluate("debate", candidate_text=debate_result.consensus, task=task, baseline_output=baseline)
            entry["debate"] = {
                "latency_ms": round(debate_time),
                "chars": len(debate_result.consensus),
                "rounds": debate_result.total_rounds,
                "overall": db_score.dimensions.overall if db_score.dimensions else 5.0,
                "delta": db_score.delta,
                "dimensions": {
                    "interface": db_score.dimensions.interface_score if db_score.dimensions else 0,
                    "functional": db_score.dimensions.functional_score if db_score.dimensions else 0,
                    "utility": db_score.dimensions.utility_score if db_score.dimensions else 0,
                },
            }
            print(f"  Debate:   {db_score.dimensions.overall if db_score.dimensions else 5.0}/10 (delta={db_score.delta}, {debate_time:.0f}ms)")
        except Exception as e:
            print(f"  Debate:   SKIPPED ({e})")
            entry["debate"] = {"error": str(e)}

        # ── Matrix ──
        try:
            t0 = time.time()
            mat_result = await matrix.solve(task)
            mat_time = (time.time() - t0) * 1000
            mat_output = mat_result.final_output
            mat_score = await gate.evaluate("matrix", candidate_text=mat_output, task=task, baseline_output=baseline)
            entry["matrix"] = {
                "latency_ms": round(mat_time),
                "chars": len(mat_output),
                "agents_used": len(mat_result.results),
                "overall": mat_score.dimensions.overall if mat_score.dimensions else 5.0,
                "delta": mat_score.delta,
                "dimensions": {
                    "interface": mat_score.dimensions.interface_score if mat_score.dimensions else 0,
                    "functional": mat_score.dimensions.functional_score if mat_score.dimensions else 0,
                    "utility": mat_score.dimensions.utility_score if mat_score.dimensions else 0,
                },
            }
            print(f"  Matrix:   {mat_score.dimensions.overall if mat_score.dimensions else 5.0}/10 (delta={mat_score.delta}, {mat_time:.0f}ms)")
        except Exception as e:
            print(f"  Matrix:   SKIPPED ({e})")
            entry["matrix"] = {"error": str(e)}

        results.append(entry)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    bl_scores = [r["baseline"]["overall"] for r in results if r["baseline"]]
    db_scores = [r["debate"]["overall"] for r in results if r["debate"] and "overall" in r["debate"]]
    db_deltas = [r["debate"]["delta"] for r in results if r["debate"] and "delta" in r["debate"]]
    mat_scores = [r["matrix"]["overall"] for r in results if r["matrix"] and "overall" in r["matrix"]]
    mat_deltas = [r["matrix"]["delta"] for r in results if r["matrix"] and "delta" in r["matrix"]]

    print(f"Baseline avg: {sum(bl_scores)/len(bl_scores):.1f}/10 ({len(bl_scores)} questions)")
    if db_scores:
        print(f"Debate avg:   {sum(db_scores)/len(db_scores):.1f}/10 (delta={sum(db_deltas)/len(db_deltas):+.1f})")
    if mat_scores:
        print(f"Matrix avg:   {sum(mat_scores)/len(mat_scores):.1f}/10 (delta={sum(mat_deltas)/len(mat_deltas):+.1f})")

    best_engine = "Baseline"
    best_score = sum(bl_scores) / len(bl_scores) if bl_scores else 0
    if db_scores and sum(db_scores) / len(db_scores) > best_score:
        best_engine = "Debate"
        best_score = sum(db_scores) / len(db_scores)
    if mat_scores and sum(mat_scores) / len(mat_scores) > best_score:
        best_engine = "Matrix"
        best_score = sum(mat_scores) / len(mat_scores)

    print(f"\nBest engine: {best_engine} ({best_score:.1f}/10)")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "questions": len(results),
        "engines_tested": ["baseline", "debate", "matrix"],
        "results": results,
        "summary": {
            "baseline_avg": round(sum(bl_scores) / len(bl_scores), 1) if bl_scores else 0,
            "debate_avg": round(sum(db_scores) / len(db_scores), 1) if db_scores else 0,
            "debate_delta": round(sum(db_deltas) / len(db_deltas), 1) if db_deltas else 0,
            "matrix_avg": round(sum(mat_scores) / len(mat_scores), 1) if mat_scores else 0,
            "matrix_delta": round(sum(mat_deltas) / len(mat_deltas), 1) if mat_deltas else 0,
            "best_engine": best_engine,
            "best_score": round(best_score, 1),
        },
    }

    report_path = PROJECT / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {report_path}")

    return report


if __name__ == "__main__":
    asyncio.run(run_benchmark())
