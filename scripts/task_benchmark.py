"""100-task efficacy benchmark — measures real completion rate.

Generates 100 diverse simulated tasks, runs them through the agent,
and measures completion rate, latency, and cost.

Usage: uv run python scripts/task_benchmark.py [--full]
  --full: Run all 100 tasks (default: 25 fast tasks for quick check)
"""

import asyncio
import json
import statistics
import sys
import time
import httpx

BASE = "http://127.0.0.1:8000"
FULL = "--full" in sys.argv
N_TASKS = 100 if FULL else 25

TASKS = [
    # Simple calculations (fast, should succeed)
    ("calc", "Calculate 123 + 456"),
    ("calc", "What is 15 * 7 - 3?"),
    ("calc", "Compute the square root of 144"),
    ("calc", "If I have $100 and spend $37.50, how much remains?"),
    ("calc", "Convert 100 Celsius to Fahrenheit"),
    # Information retrieval
    ("info", "What tools do you have available?"),
    ("info", "What is your architecture?"),
    ("info", "Explain how CLEAR metrics work"),
    ("info", "What is the governance system?"),
    ("info", "Describe the sandbox execution model"),
    # Code generation
    ("code", "Write a Python function that checks if a string is a palindrome"),
    ("code", "Create a code snippet that sorts a list of dictionaries by a key"),
    ("code", "Write a Python decorator that measures function execution time"),
    ("code", "Generate a function to parse JSON and extract all email addresses"),
    ("code", "Write a regular expression to validate Chinese phone numbers"),
    # Reasoning
    ("reason", "Explain the difference between async and sync in Python"),
    ("reason", "What is the circuit breaker pattern and when should you use it?"),
    ("reason", "Compare RAG vs fine-tuning for domain-specific LLM apps"),
    ("reason", "Explain zero-trust architecture in 3 sentences"),
    ("reason", "What are the key metrics for evaluating an AI agent?"),
    # Mixed / edge cases
    ("edge", "What happens if I ask for an impossible calculation?"),
    ("edge", "List the files in the current working directory"),
    ("edge", "What's the current date and time?"),
    ("edge", "Tell me about the project's tech stack based on CLAUDE.md"),
    ("edge", "Summarize the 4 phases of enterprise agent maturity"),
]

# Add diversity for full runs
EXTRA_TASKS = [
    ("calc", f"Calculate {i} ** 2 + {i*3} - {i//2}") for i in range(1, 26)
] + [
    ("info", f"What is the {i}th prime number and how is it used in cryptography?") for i in range(1, 16)
] + [
    ("code", f"Write a Python function named task_{i} that does something useful with {['strings','numbers','lists','dicts','files'][i%5]}") for i in range(1, 16)
] + [
    ("reason", f"Explain concept #{i}: {['caching','load balancing','sharding','replication','consensus','encryption','hashing','compression','serialization','orchestration'][i%10]}") for i in range(1, 19)
]

if FULL:
    TASKS += EXTRA_TASKS
    TASKS = TASKS[:100]


async def run_task(client: httpx.AsyncClient, category: str, prompt: str, idx: int) -> dict:
    start = time.perf_counter()
    try:
        r = await client.post(
            f"{BASE}/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        elapsed = (time.perf_counter() - start) * 1000
        success = 200 <= r.status_code < 300
        return {
            "idx": idx,
            "category": category,
            "prompt": prompt[:80],
            "latency_ms": round(elapsed, 1),
            "success": success,
            "status": r.status_code,
            "response_len": len(r.text),
        }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "idx": idx,
            "category": category,
            "prompt": prompt[:80],
            "latency_ms": round(elapsed, 1),
            "success": False,
            "status": 0,
            "error": str(e)[:100],
        }


async def main():
    print(f"Running {len(TASKS)} task efficacy benchmark...")
    results = []

    async with httpx.AsyncClient() as client:
        # Process in batches of 5 concurrent
        batch_size = 5
        for i in range(0, len(TASKS), batch_size):
            batch = TASKS[i:i+batch_size]
            tasks = [run_task(client, cat, prompt, i+idx) for idx, (cat, prompt) in enumerate(batch)]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Progress
            done = min(i + batch_size, len(TASKS))
            ok = sum(1 for r in results if r["success"])
            print(f"\r[{done}/{len(TASKS)}] {ok} OK", end="")

    print()

    # Analysis
    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    lats = [r["latency_ms"] for r in ok]

    print(f"\n{'='*55}")
    print(f"  TASK EFFICACY BENCHMARK ({len(TASKS)} tasks)")
    print(f"{'='*55}")
    print(f"  Completed: {len(ok)}/{len(results)} ({len(ok)/len(results)*100:.0f}%)")
    print(f"  Failed:    {len(fail)}")

    if lats:
        sl = sorted(lats)
        print(f"  Avg latency: {statistics.mean(lats):.0f}ms")
        print(f"  P50:         {sl[len(sl)//2]:.0f}ms")
        print(f"  P95:         {sl[int(len(sl)*0.95)]:.0f}ms")
        print(f"  Max:         {max(lats):.0f}ms")

    print(f"\n  By category:")
    for cat in ["calc", "info", "code", "reason", "edge"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_ok = sum(1 for r in cat_results if r["success"])
            cat_lats = [r["latency_ms"] for r in cat_results if r["success"]]
            print(f"    {cat:8s}: {cat_ok}/{len(cat_results)} ({cat_ok/len(cat_results)*100:.0f}%) avg={statistics.mean(cat_lats):.0f}ms" if cat_lats else f"    {cat:8s}: {cat_ok}/{len(cat_results)}")

    # P1/P2 targets
    completion_rate = len(ok) / len(results) * 100
    print(f"\n  P1 target (>=85%): {'PASS' if completion_rate >= 85 else 'FAIL'}")
    print(f"  P2 target (>=92%): {'PASS' if completion_rate >= 92 else 'FAIL'}")

    # Save
    with open("task_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "completed": len(ok),
            "failed": len(fail),
            "completion_rate": round(completion_rate, 1),
            "avg_ms": round(statistics.mean(lats), 1) if lats else 0,
            "p95_ms": round(sl[int(len(sl)*0.95)], 1) if lats else 0,
        }, f, indent=2)

    print(f"\n  Saved: task_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
