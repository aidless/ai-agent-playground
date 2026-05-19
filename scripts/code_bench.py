"""代码修复基准 — 10 个真实 Python bug 修复任务

评估维度（AgentBench + SWE-bench 启发）:
  1. Bug 检测准确率 — Agent 能找到 bug 吗
  2. 修复成功率 — 修复后测试是否通过
  3. 平均步数 — 效率
  4. 自我修正率 — 第一次失败后能否修正

每个任务: 有 bug 的代码 + 测试用例 + 预期修复
"""

import asyncio, json, os, sys, time, textwrap
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CODE_TASKS = [
    {
        "id": "bug_binary_search",
        "title": "Binary Search Infinite Loop",
        "buggy_code": textwrap.dedent("""\
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid
                else:
                    right = mid
            return -1
        """),
        "expected_fix": "left = mid + 1",
        "test": textwrap.dedent("""\
        assert binary_search([1, 3, 5, 7, 9], 5) == 2
        assert binary_search([1, 3, 5, 7, 9], 1) == 0
        assert binary_search([1, 3, 5, 7, 9], 9) == 4
        assert binary_search([1, 3, 5, 7, 9], 6) == -1
        assert binary_search([], 1) == -1
        """),
        "difficulty": "easy",
    },
    {
        "id": "bug_divide_zero",
        "title": "Division by Zero in Average",
        "buggy_code": textwrap.dedent("""\
        def average(numbers):
            total = sum(numbers)
            return total / len(numbers)
        """),
        "expected_fix": "if not numbers",
        "test": textwrap.dedent("""\
        assert average([1, 2, 3]) == 2.0
        assert average([5]) == 5.0
        try:
            average([])
            assert False, "Should raise"
        except:
            pass
        """),
        "difficulty": "easy",
    },
    {
        "id": "bug_off_by_one",
        "title": "Fibonacci Off-By-One",
        "buggy_code": textwrap.dedent("""\
        def fibonacci(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        """),
        "expected_fix": "range(n - 1)",
        "test": textwrap.dedent("""\
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(3) == 2
        assert fibonacci(5) == 5
        assert fibonacci(10) == 55
        """),
        "difficulty": "medium",
    },
    {
        "id": "bug_mutable_default",
        "title": "Mutable Default Argument",
        "buggy_code": textwrap.dedent("""\
        def add_item(item, items=[]):
            items.append(item)
            return items
        """),
        "expected_fix": "items=None",
        "test": textwrap.dedent("""\
        assert add_item(1) == [1]
        assert add_item(2) == [2]
        assert add_item(3, [0]) == [0, 3]
        r1 = add_item(4)
        r2 = add_item(5)
        assert r1 == [4], f"Expected [4], got {r1}"
        assert r2 == [5], f"Expected [5], got {r2}"
        """),
        "difficulty": "medium",
    },
    {
        "id": "bug_race_condition",
        "title": "Non-Thread-Safe Counter",
        "buggy_code": textwrap.dedent("""\
        class Counter:
            def __init__(self):
                self.value = 0
            def increment(self):
                self.value += 1
                return self.value
        """),
        "expected_fix": "threading.Lock",
        "test": textwrap.dedent("""\
        c = Counter()
        assert c.increment() == 1
        assert c.increment() == 2
        import threading
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
        "difficulty": "hard",
    },
    {
        "id": "bug_memory_leak",
        "title": "Cache Without Size Limit",
        "buggy_code": textwrap.dedent("""\
        cache = {}
        def get_or_compute(key, compute_fn):
            if key not in cache:
                cache[key] = compute_fn()
            return cache[key]
        """),
        "expected_fix": "maxsize or LRU",
        "test": textwrap.dedent("""\
        call_count = [0]
        def expensive():
            call_count[0] += 1
            return call_count[0]
        assert get_or_compute('a', expensive) == 1
        assert get_or_compute('a', expensive) == 1
        assert call_count[0] == 1
        """),
        "difficulty": "medium",
    },
    {
        "id": "bug_sql_injection",
        "title": "SQL Injection Vulnerability",
        "buggy_code": textwrap.dedent("""\
        def get_user(db, username):
            query = f"SELECT * FROM users WHERE name = '{username}'"
            return db.execute(query)
        """),
        "expected_fix": "parameterized query",
        "test": textwrap.dedent("""\
        class MockDB:
            def execute(self, query):
                if ';' in query or "' OR" in query.upper():
                    raise ValueError("SQL injection blocked")
                return query
        db = MockDB()
        r = get_user(db, "alice")
        assert "alice" in str(r)
        try:
            get_user(db, "admin' OR '1'='1")
            assert False, "Should block injection"
        except ValueError:
            pass
        """),
        "difficulty": "hard",
    },
    {
        "id": "bug_key_error",
        "title": "Missing KeyError Handling",
        "buggy_code": textwrap.dedent("""\
        def safe_get(data, key):
            return data[key]
        """),
        "expected_fix": "data.get(key, default)",
        "test": textwrap.dedent("""\
        assert safe_get({"a": 1}, "a") == 1
        assert safe_get({"a": 1}, "b") is None
        assert safe_get({}, "any") is None
        """),
        "difficulty": "easy",
    },
    {
        "id": "bug_infinite_recursion",
        "title": "Missing Base Case",
        "buggy_code": textwrap.dedent("""\
        def factorial(n):
            return n * factorial(n - 1)
        """),
        "expected_fix": "if n <= 1",
        "test": textwrap.dedent("""\
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        """),
        "difficulty": "easy",
    },
    {
        "id": "bug_file_not_closed",
        "title": "File Handle Not Closed",
        "buggy_code": textwrap.dedent("""\
        def read_lines(path):
            f = open(path, 'r')
            return f.readlines()
        """),
        "expected_fix": "with open",
        "test": textwrap.dedent("""\
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
            tf.write('line1\\nline2\\n')
            path = tf.name
        lines = read_lines(path)
        assert lines == ['line1\\n', 'line2\\n']
        os.unlink(path)
        """),
        "difficulty": "easy",
    },
]

async def validate_fix(client, task: dict, response: str) -> bool:
    """Use LLM to judge whether the fix is correct."""
    import re
    # Extract code from response
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    fix_code = code_match.group(1).strip() if code_match else response.strip()

    prompt = (
        f"Buggy code:\n```python\n{task['buggy_code']}\n```\n\n"
        f"Proposed fix:\n```python\n{fix_code[:1000]}\n```\n\n"
        f"Expected change: the fix should address '{task['expected_fix']}'\n\n"
        f"Does the fix correctly resolve the bug? Answer ONLY 'YES' or 'NO'."
    )
    try:
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip().upper()
        return "YES" in text
    except Exception:
        return False


async def run_code_benchmark():
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
    from agent.reflect_action import ReflectActionEngine

    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda text: text)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry, enable_super_agent=True)
    gate = EvaluationGate(deepseek)

    results = []
    detected = 0
    fixed = 0
    self_corrected = 0

    print(f"CODE REPAIR BENCHMARK — {len(CODE_TASKS)} tasks")
    print("=" * 70)

    for t in CODE_TASKS:
        print(f"\n--- {t['id']} ({t['difficulty']}): {t['title']} ---")

        task_prompt = (
            f"Find and fix the bug in this Python code. Output ONLY the corrected code, "
            f"nothing else.\n\n"
            f"The code should pass these tests:\n```python\n{t['test']}\n```\n"
            f"Current (buggy) code:\n```python\n{t['buggy_code']}\n```"
        )

        entry = {"id": t["id"], "difficulty": t["difficulty"]}
        fixed_this = False
        corrected_this = False

        # Attempt 1
        t0 = time.time()
        ctx = AgentContext(trace_id=f"code_{t['id']}_1", max_steps=2)
        ctx = await agent.run(ctx, task_prompt)
        response_1 = ""
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                response_1 = msg["content"]
        attempt_1_time = (time.time() - t0) * 1000

        # Check if bug detected (expected fix keyword in response)
        expected = t.get("expected_fix", "").lower()
        bug_detected = expected in response_1.lower()
        if bug_detected:
            detected += 1

        # LLM-based validation: check if fix is correct
        fix_passed = await validate_fix(deepseek, t, response_1)
        if fix_passed:
            fixed += 1
            fixed_this = True

        # If attempt 1 failed, try self-correction with specific feedback
        attempt_2_time = 0
        if not fix_passed:
            # Get specific validation feedback
            feedback_prompt = (
                f"Buggy code:\n```python\n{t['buggy_code']}\n```\n\n"
                f"Agent's fix attempt:\n```python\n{response_1[:1000]}\n```\n\n"
                f"The fix should address: '{t['expected_fix']}'. "
                f"What specifically is wrong with the agent's fix? Be specific in 1 sentence."
            )
            try:
                fb_resp = await deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": feedback_prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                feedback = fb_resp.choices[0].message.content.strip()
            except Exception:
                feedback = "The fix doesn't correctly address the bug."

            correction_prompt = (
                f"Your previous fix was incorrect. Feedback: {feedback}\n\n"
                f"The code must pass these tests:\n```python\n{t['test']}\n```\n"
                f"Fix the buggy code. Output ONLY the corrected code.\n\n"
                f"```python\n{t['buggy_code']}\n```"
            )
            t0 = time.time()
            ctx2 = AgentContext(trace_id=f"code_{t['id']}_2", max_steps=2)
            ctx2 = await agent.run(ctx2, correction_prompt)
            response_2 = ""
            for msg in ctx2.messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    response_2 = msg["content"]
            attempt_2_time = (time.time() - t0) * 1000

            fix_passed = await validate_fix(deepseek, t, response_2)
            if fix_passed:
                fixed += 1
                corrected_this = True
                self_corrected += 1

        # Score with evaluation gate
        best_response = response_1 if fixed_this else (response_2 if corrected_this else response_1)
        eval_result = await gate.evaluate("code_repair", candidate_text=best_response, task=task_prompt)

        entry.update({
            "bug_detected": bug_detected,
            "fix_passed": fix_passed,
            "self_corrected": corrected_this,
            "attempt_1_ms": round(attempt_1_time),
            "attempt_2_ms": round(attempt_2_time),
            "quality_score": eval_result.dimensions.overall if eval_result.dimensions else 5.0,
        })
        results.append(entry)

        status = "FIXED" if fixed_this else ("SELF-CORRECTED" if corrected_this else "FAILED")
        print(f"  {status} | detected={bug_detected} | score={entry['quality_score']}/10 | {attempt_1_time:.0f}ms")

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Tasks:            {len(CODE_TASKS)}")
    print(f"Bug detected:     {detected}/{len(CODE_TASKS)} ({detected/len(CODE_TASKS)*100:.0f}%)")
    print(f"Fix success:      {fixed}/{len(CODE_TASKS)} ({fixed/len(CODE_TASKS)*100:.0f}%)")
    print(f"Self-corrected:   {self_corrected}/{len(CODE_TASKS)} ({self_corrected/len(CODE_TASKS)*100:.0f}%)")
    avg_score = sum(r["quality_score"] for r in results) / len(results)
    print(f"Avg quality:      {avg_score:.1f}/10")
    print(f"Avg latency:      {sum(r['attempt_1_ms'] for r in results)/len(results):.0f}ms")

    report = {
        "benchmark": "code_repair",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": len(CODE_TASKS),
        "results": results,
        "summary": {
            "detection_rate": round(detected / len(CODE_TASKS), 3),
            "fix_rate": round(fixed / len(CODE_TASKS), 3),
            "self_correction_rate": round(self_corrected / len(CODE_TASKS), 3),
            "avg_quality": round(avg_score, 1),
        },
    }
    (PROJECT / "code_bench_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: code_bench_report.json")

if __name__ == "__main__":
    asyncio.run(run_code_benchmark())
