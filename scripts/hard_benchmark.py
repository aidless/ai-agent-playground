"""Hard Benchmark — questions where single models commonly make mistakes.

Tests baseline (single DeepSeek) vs debate (DeepSeek vs Qwen2.5).
Uses questions with objective correctness to measure real improvement.

Usage: uv run python scripts/hard_benchmark.py
"""

import asyncio, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import dotenv
from openai import AsyncOpenAI

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
dotenv.load_dotenv(PROJECT / ".env")

HARD_QUESTIONS = [
    {
        "id": "hard_logic",
        "domain": "logic",
        "task": "There are 5 houses in a row, each a different color. The owner of each house has a different nationality, drinks a different beverage, smokes a different brand of cigar, and keeps a different pet. Given these clues, who owns the zebra? 1. The Norwegian lives in the first house. 2. The Englishman lives in the red house. 3. The green house is immediately to the left of the white house. 4. The person who drinks coffee lives in the green house. 5. The person who smokes Pall Mall keeps birds. 6. The owner of the yellow house smokes Dunhill. 7. The man living in the center house drinks milk. 8. The man who smokes Blends lives next to the one who keeps cats. 9. The Norwegian lives next to the blue house. 10. The person who keeps horses lives next to the one who smokes Dunhill. 11. The man who smokes Blue Master drinks beer. 12. The German smokes Prince. 13. The Swede keeps dogs. 14. The man who smokes Blends has a neighbor who drinks water. Solve step by step.",
        "expected_answer": "German",
    },
    {
        "id": "hard_code_bug",
        "domain": "coding",
        "task": "This Python function has a bug. Find it and explain:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1\n```",
        "expected_answer": "infinite loop",
    },
    {
        "id": "hard_math_edge",
        "domain": "math",
        "task": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your work. (This is a famous cognitive reflection test — be careful.)",
        "expected_answer": "0.05",
    },
    {
        "id": "hard_security_bypass",
        "domain": "security",
        "task": "Analyze this authentication code for security vulnerabilities:\n\n```python\ndef login(username, password):\n    query = f\"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'\"\n    result = db.execute(query)\n    if result:\n        token = jwt.encode({'user': username}, 'secret123', algorithm='HS256')\n        return {'token': token}\n    return {'error': 'Invalid credentials'}\n```\n\nList ALL security issues.",
        "expected_answer": "SQL injection + hardcoded secret + no password hashing + no expiry + no HTTPS",
    },
    {
        "id": "hard_recursion",
        "domain": "algorithms",
        "task": "Explain why the following recursive Fibonacci implementation is problematic and provide an optimized version:\n\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```\n\nAlso: what's the time complexity, and what value of n would you consider the limit for this implementation?",
        "expected_answer": "O(2^n)",
    },
]


async def main():
    deepseek = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    ollama = None
    try:
        ollama = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    except Exception:
        pass

    from agent.eval_gate import EvaluationGate
    from agent.debate import DebateEngine
    from agent.async_core import AsyncAgent
    from agent.state import AgentContext
    from agent.tools.registry import ToolRegistry

    gate = EvaluationGate(deepseek)
    debate_eng = DebateEngine(deepseek, ollama or deepseek, deepseek)
    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda text: text)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry)

    results = []
    bl_scores = []
    db_scores = []
    bl_pass = 0
    db_pass = 0

    print(f"HARD BENCHMARK: {len(HARD_QUESTIONS)} challenging questions")
    print(f"{'='*70}")

    for q in HARD_QUESTIONS:
        print(f"\n--- {q['id']} ({q['domain']}) ---")
        task = q["task"]

        # Baseline
        t0 = time.time()
        ctx = AgentContext(trace_id=f"hard_bl_{q['id']}", max_steps=2)
        ctx = await agent.run(ctx, task)
        baseline = ""
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                baseline = msg["content"]
        bl_time = (time.time() - t0) * 1000
        bl_eval = await gate.evaluate("hard_baseline", candidate_text=baseline, task=task)
        bl_overall = bl_eval.dimensions.overall if bl_eval.dimensions else 5.0

        # Check if baseline got the expected answer
        expected = q.get("expected_answer", "").lower()
        bl_correct = expected in baseline.lower() if expected else None

        print(f"  Baseline: {bl_overall}/10 ({bl_time:.0f}ms) correct={bl_correct}")

        # Debate
        try:
            t0 = time.time()
            db_result = await debate_eng.debate_process_centric(
                task, "deepseek-chat", "qwen2.5:7b" if ollama else "deepseek-chat"
            )
            db_time = (time.time() - t0) * 1000
            db_eval = await gate.evaluate("hard_debate", candidate_text=db_result.consensus, task=task, baseline_output=baseline)
            db_overall = db_eval.dimensions.overall if db_eval.dimensions else 5.0

            db_correct = expected in db_result.consensus.lower() if expected else None
            delta_correct = (db_correct and not bl_correct) if (db_correct is not None and bl_correct is not None) else None

            print(f"  Debate:   {db_overall}/10 ({db_time:.0f}ms) correct={db_correct} | delta={db_eval.delta:+} fix={delta_correct}")
        except Exception as e:
            print(f"  Debate:   SKIPPED ({e})")
            db_overall = bl_overall
            db_correct = bl_correct
            delta_correct = None
            db_eval = None

        bl_scores.append(bl_overall)
        db_scores.append(db_overall)
        if bl_correct:
            bl_pass += 1
        if db_correct:
            db_pass += 1

        results.append({
            "id": q["id"],
            "domain": q["domain"],
            "baseline_score": bl_overall,
            "baseline_correct": bl_correct,
            "debate_score": db_overall,
            "debate_correct": db_correct,
            "debate_fixed_baseline_error": delta_correct,
            "debate_delta": db_eval.delta if db_eval else 0,
        })

    # Summary
    print(f"\n{'='*70}")
    print("HARD BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Baseline avg: {sum(bl_scores)/len(bl_scores):.1f}/10 | correct: {bl_pass}/{len(HARD_QUESTIONS)}")
    print(f"Debate avg:   {sum(db_scores)/len(db_scores):.1f}/10 | correct: {db_pass}/{len(HARD_QUESTIONS)}")

    fixes = sum(1 for r in results if r.get("debate_fixed_baseline_error"))
    print(f"\nDebate fixed baseline errors: {fixes}/{len(HARD_QUESTIONS)}")
    print(f"Score delta: {sum(db_scores)/len(db_scores) - sum(bl_scores)/len(bl_scores):+.1f}")

    for r in results:
        status = "✅ FIXED" if r.get("debate_fixed_baseline_error") else ("✅ BOTH" if (r["baseline_correct"] and r["debate_correct"]) else "⚠ SAME" if (r["baseline_correct"] == r["debate_correct"]) else "❌ WORSE")
        print(f"  {r['id']}: bl_correct={r['baseline_correct']} db_correct={r['debate_correct']} {status}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "hard_benchmark",
        "questions": len(HARD_QUESTIONS),
        "results": results,
        "summary": {
            "baseline_avg": round(sum(bl_scores)/len(bl_scores), 1),
            "debate_avg": round(sum(db_scores)/len(db_scores), 1),
            "baseline_correct": f"{bl_pass}/{len(HARD_QUESTIONS)}",
            "debate_correct": f"{db_pass}/{len(HARD_QUESTIONS)}",
            "errors_fixed_by_debate": fixes,
        },
    }
    (PROJECT / "hard_benchmark_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: hard_benchmark_report.json")


if __name__ == "__main__":
    asyncio.run(main())
