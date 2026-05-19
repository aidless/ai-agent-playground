"""多智能体协作基准 — SAGE-inspired multi-agent benchmark

评估 Crew + Debate + Matrix 在协作任务上的表现:
  1. Multi-perspective quality
  2. Role adherence
  3. Collaboration efficiency
"""

import asyncio, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

AGENT_TASKS = [
    {
        "id": "mab_design_review",
        "task": "Design a REST API for a todo app and review it for security issues. Provide both the design AND the security review.",
        "roles_needed": ["designer", "reviewer"],
        "evaluate": "completeness, security awareness, API design quality",
    },
    {
        "id": "mab_code_and_test",
        "task": "Write a function to validate email addresses AND write unit tests for it. Include both implementation and tests.",
        "roles_needed": ["developer", "tester"],
        "evaluate": "code correctness, test coverage, edge cases",
    },
    {
        "id": "mab_arch_decision",
        "task": "Compare SQL vs NoSQL for a social media app. Provide pros/cons and a final recommendation with reasoning.",
        "roles_needed": ["architect", "reasoner"],
        "evaluate": "depth of analysis, practical reasoning, clear recommendation",
    },
    {
        "id": "mab_security_audit",
        "task": "Audit this login function for vulnerabilities and write the corrected version:\n```python\ndef login(user, pwd):\n    query = f\"SELECT * FROM users WHERE name='{user}' AND pwd='{pwd}'\"\n    result = db.execute(query)\n    return result is not None\n```",
        "roles_needed": ["reviewer", "developer"],
        "evaluate": "vulnerability identification, correct fix, explanation quality",
    },
    {
        "id": "mab_scale_plan",
        "task": "Your web app has 100K users and is slowing down. Describe a step-by-step scaling plan: caching, database, load balancing, monitoring.",
        "roles_needed": ["architect", "developer"],
        "evaluate": "practical planning, technical depth, prioritization",
    },
]

async def run_multi_agent_benchmark():
    import dotenv
    from openai import AsyncOpenAI
    dotenv.load_dotenv(PROJECT / ".env")

    deepseek = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry
    from agent.eval_gate import EvaluationGate
    from agent.matrix import AgentMatrix, MatrixAgentProfile

    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda text: text)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry)
    gate = EvaluationGate(deepseek)

    # Multi-agent matrix
    matrix = AgentMatrix()
    matrix.add_agent(MatrixAgentProfile("ds-reasoner", "DeepSeek Reasoner", "reasoner", "deepseek-chat", deepseek))
    matrix.add_agent(MatrixAgentProfile("ds-coder", "DeepSeek Coder", "coder", "deepseek-chat", deepseek))
    matrix.add_agent(MatrixAgentProfile("ds-reviewer", "DeepSeek Reviewer", "reviewer", "deepseek-chat", deepseek))

    results = []

    print(f"MULTI-AGENT BENCHMARK — {len(AGENT_TASKS)} tasks")
    print("=" * 70)

    for t in AGENT_TASKS:
        print(f"\n--- {t['id']} ---")
        print(f"Task: {t['task'][:80]}...")

        entry = {"id": t["id"], "roles_needed": t["roles_needed"]}

        # Baseline: single agent
        t0 = time.time()
        ctx = AgentContext(trace_id=f"mab_bl_{t['id']}", max_steps=3)
        ctx = await agent.run(ctx, t["task"])
        baseline = ""
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                baseline = msg["content"]
        bl_time = (time.time() - t0) * 1000

        # Matrix: multi-agent routing
        t0 = time.time()
        mat = await matrix.solve(t["task"])
        mat_time = (time.time() - t0) * 1000
        mat_output = mat.final_output
        agents_used = len(mat.results)

        # Evaluate both
        bl_score = await gate.evaluate("multi_baseline", candidate_text=baseline, task=t["task"])
        mat_score = await gate.evaluate("multi_matrix", candidate_text=mat_output, task=t["task"], baseline_output=baseline)

        bl_overall = bl_score.dimensions.overall if bl_score.dimensions else 5.0
        mat_overall = mat_score.dimensions.overall if mat_score.dimensions else 5.0
        delta = mat_score.delta

        entry.update({
            "baseline_score": bl_overall,
            "matrix_score": mat_overall,
            "matrix_delta": delta,
            "agents_used": agents_used,
            "baseline_ms": round(bl_time),
            "matrix_ms": round(mat_time),
        })
        results.append(entry)

        winner = "Matrix" if mat_overall > bl_overall else ("Tie" if mat_overall == bl_overall else "Baseline")
        print(f"  Baseline: {bl_overall}/10 ({bl_time:.0f}ms) | Matrix: {mat_overall}/10 ({mat_time:.0f}ms, {agents_used} agents)")
        print(f"  Winner: {winner} (delta={delta:+.1f})")

    # Summary
    bl_avg = sum(r["baseline_score"] for r in results) / len(results)
    mat_avg = sum(r["matrix_score"] for r in results) / len(results)
    mat_wins = sum(1 for r in results if r["matrix_score"] > r["baseline_score"])
    avg_agents = sum(r["agents_used"] for r in results) / len(results)

    print(f"\n{'='*70}")
    print("MULTI-AGENT BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Tasks:              {len(AGENT_TASKS)}")
    print(f"Baseline avg:       {bl_avg:.1f}/10")
    print(f"Matrix avg:         {mat_avg:.1f}/10")
    print(f"Avg delta:          {mat_avg - bl_avg:+.1f}")
    print(f"Matrix wins:        {mat_wins}/{len(AGENT_TASKS)}")
    print(f"Avg agents/task:    {avg_agents:.1f}")
    print(f"Avg baseline ms:    {sum(r['baseline_ms'] for r in results)/len(results):.0f}")
    print(f"Avg matrix ms:      {sum(r['matrix_ms'] for r in results)/len(results):.0f}")

    report = {
        "benchmark": "multi_agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": len(AGENT_TASKS),
        "results": results,
        "summary": {
            "baseline_avg": round(bl_avg, 1),
            "matrix_avg": round(mat_avg, 1),
            "delta": round(mat_avg - bl_avg, 1),
            "matrix_wins": mat_wins,
            "total_tasks": len(AGENT_TASKS),
        },
    }
    (PROJECT / "multi_agent_bench_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: multi_agent_bench_report.json")

if __name__ == "__main__":
    asyncio.run(run_multi_agent_benchmark())
