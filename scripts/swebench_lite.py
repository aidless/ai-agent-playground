"""SWE-bench Lite Adapter — standard benchmark for code repair agents.

Evaluates the agent on realistic GitHub-issue style tasks with:
  - Multi-file projects (not single functions)
  - Issue descriptions (like GitHub issues)
  - Test-based evaluation (like SWE-bench)
  - Resolve rate metric (standard SWE-bench metric)

These tasks are inspired by SWE-bench Lite's methodology but self-contained.

Usage: uv run python scripts/swebench_lite.py
"""

import asyncio, json, os, sys, time, textwrap
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SWE_TASKS = [
    {
        "id": "swe_fix_null_pointer",
        "repo": "user-service",
        "issue": "Bug: NullPointerException when user email is missing",
        "description": textwrap.dedent("""\
        Users reported a crash when calling GET /users/{id} for users without email addresses.
        The get_user() function doesn't handle None values in the email field.
        Return an empty string "" instead of crashing.
        Expected behavior: GET /users/45 returns {"name":"Bob","email":""}
        Current behavior: crashes with AttributeError
        """),
        "before": textwrap.dedent("""\
        class UserService:
            def get_user(self, user_id: int) -> dict:
                data = self.db.query(f"SELECT * FROM users WHERE id={user_id}")
                if not data:
                    raise ValueError(f"User {user_id} not found")
                return {
                    "name": data["name"],
                    "email": data["email"].strip(),
                }
        """),
        "test": textwrap.dedent("""\
        class UserService:
            def __init__(self):
                self._db = {}
            def db(self):
                return type('DB',(),{'query':lambda q:self._db.get(45)})()
        svc = UserService()
        svc._db[45] = {"name":"Bob","email":None}
        u = svc.get_user(45)
        assert u["name"] == "Bob"
        assert u["email"] == ""
        svc._db[45] = {"name":"Alice","email":"alice@test.com"}
        u = svc.get_user(45)
        assert u["email"] == "alice@test.com"
        """),
        "expected_fix": ".get(\"email\",\"\") or .strip() if",
    },
    {
        "id": "swe_fix_race_condition",
        "repo": "counter-service",
        "issue": "Race condition: counter returns wrong value under concurrent access",
        "description": textwrap.dedent("""\
        The Counter.increment() method is not thread-safe. Under concurrent access from 100 threads,
        the final count is less than 100 instead of exactly 100. Add thread safety.
        """),
        "before": textwrap.dedent("""\
        class Counter:
            def __init__(self):
                self.value = 0
            def increment(self):
                self.value += 1
                return self.value
        """),
        "test": textwrap.dedent("""\
        import threading
        c = Counter()
        assert c.increment() == 1
        assert c.increment() == 2
        # Concurrent test
        c2 = Counter()
        threads = []
        for _ in range(100):
            t = threading.Thread(target=c2.increment)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert c2.value == 100, f"Expected 100, got {c2.value}"
        """),
        "expected_fix": "threading.Lock",
    },
    {
        "id": "swe_fix_sql_injection",
        "repo": "login-service",
        "issue": "Security: SQL injection vulnerability in user lookup",
        "description": textwrap.dedent("""\
        The get_user_by_name() function directly interpolates user input into SQL query.
        An attacker can bypass authentication with: admin' OR '1'='1
        Convert to parameterized query.
        """),
        "before": textwrap.dedent("""\
        class LoginService:
            def get_user_by_name(self, username: str) -> dict:
                query = f"SELECT * FROM users WHERE name='{username}'"
                return self.db.execute(query)
        """),
        "test": textwrap.dedent("""\
        class MockDB:
            def __init__(self):
                self.last_query = ""
            def execute(self, query, params=None):
                self.last_query = query
                if params:
                    return {"name": params[0], "role": "user"}
                return None
        svc = LoginService()
        mock = MockDB()
        svc.db = mock
        # Normal query
        r = svc.get_user_by_name("alice")
        assert r is not None
        assert r["name"] == "alice"
        # Injection attempt should be blocked by parameterization
        # The mock doesn't check SQL grammar, just verifies no string interpolation
        """),
        "expected_fix": "? AND params",
    },
]

async def run_swebench():
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

    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda t: t)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry, enable_super_agent=True)

    results = []
    resolved = 0

    print("SWE-BENCH LITE ADAPTER")
    print(f"{len(SWE_TASKS)} tasks")
    print("=" * 60)

    for t in SWE_TASKS:
        print(f"\n--- {t['id']} ({t['repo']}) ---")

        task_prompt = (
            f"Repository: {t['repo']}\n"
            f"Issue: {t['issue']}\n\n"
            f"Description:\n{t['description']}\n\n"
            f"Current code:\n```python\n{t['before']}\n```\n\n"
            f"Fix the bug. Output ONLY the corrected code."
        )

        entry = {"id": t["id"], "repo": t["repo"]}

        t0 = time.time()
        ctx = AgentContext(trace_id=f"swe_{t['id']}", max_steps=3)
        ctx = await agent.run(ctx, task_prompt)
        response = ""
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
        attempt_time = (time.time() - t0) * 1000

        # Extract fixed code
        import re
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        fixed_code = code_match.group(1).strip() if code_match else response.strip()

        # Check if expected fix pattern is present
        expected = t.get("expected_fix", "").lower()
        fix_detected = expected in fixed_code.lower() or expected in response.lower()

        # Run test
        fix_passed = False
        try:
            exec(fixed_code, globals())
            exec(t["test"])
            fix_passed = True
        except Exception:
            pass

        if fix_passed:
            resolved += 1

        blind_apply = False
        try:
            exec(t["before"], globals())
            exec(t["test"])
            blind_apply = True
        except Exception:
            pass

        entry.update({
            "fix_detected": fix_detected,
            "fix_passed": fix_passed,
            "blind_apply_passes": blind_apply,
            "latency_ms": round(attempt_time),
        })
        results.append(entry)

        status = "RESOLVED" if fix_passed else "FAILED"
        print(f"  {status} | detected={fix_detected} | {attempt_time:.0f}ms")

    # SWE-bench style metrics
    resolve_rate = resolved / len(SWE_TASKS)
    print(f"\n{'=' * 60}")
    print("SWE-BENCH LITE RESULTS")
    print(f"{'=' * 60}")
    print(f"Tasks:        {len(SWE_TASKS)}")
    print(f"Resolved:     {resolved}/{len(SWE_TASKS)}")
    print(f"Resolve Rate: {resolve_rate:.1%}")
    print(f"Avg Latency:  {sum(r['latency_ms'] for r in results)/len(results):.0f}ms")

    report = {
        "benchmark": "swebench_lite",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": len(SWE_TASKS),
        "resolved": resolved,
        "resolve_rate": round(resolve_rate, 3),
        "results": results,
    }
    path = PROJECT / "swebench_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {path}")

    return report


if __name__ == "__main__":
    asyncio.run(run_swebench())
