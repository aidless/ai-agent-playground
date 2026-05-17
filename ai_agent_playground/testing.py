"""Testing Framework — Agent测试。"""

import unittest
from dataclasses import dataclass, field
from typing import Any


class MockLLM:
    def __init__(self):
        self._responses: dict[str, str] = {}
        self._call_history: list[dict] = []
        self._call_count = 0

    def add_response(self, prompt_pattern: str, response: str):
        self._responses[prompt_pattern] = response

    def send(self, messages: list, **kwargs) -> str:
        self._call_count += 1
        full_prompt = "\n".join(m.get("content", "") for m in messages)
        self._call_history.append({"messages": messages, "kwargs": kwargs, "call_number": self._call_count})
        for pattern, response in self._responses.items():
            if pattern in full_prompt:
                return response
        return "[Mock] Default response"

    def send_stream(self, messages: list, **kwargs):
        response = self.send(messages, **kwargs)
        for word in response.split():
            yield word + " "

    def get_call_history(self) -> list[dict]:
        return self._call_history.copy()

    def reset(self):
        self._call_history.clear()
        self._call_count = 0


def create_test_agent(agent_class, mock_llm: "MockLLM | None" = None):
    from ai_agent_playground.config import BaseAgentConfig
    config = BaseAgentConfig()
    agent = agent_class(config)
    if mock_llm:
        agent.llm = mock_llm
    return agent


def run_agent_test(agent, input_data: Any, expected_contains: str | None = None) -> dict:
    result = agent.run(input_data)
    passed = True
    reason = ""
    if expected_contains:
        if expected_contains not in str(result):
            passed = False
            reason = f"Expected '{expected_contains}' in result"
    return {"passed": passed, "reason": reason, "input": input_data, "result": result}


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MockLLM()

    def assertAgentResponse(self, agent, input_data: Any, expected_contains: str | None = None):
        result = agent.run(input_data)
        if expected_contains:
            self.assertIn(expected_contains, str(result))
        return result

    def assertNoError(self, agent, input_data: Any):
        try:
            agent.run(input_data)
        except Exception as e:
            self.fail(f"Agent raised exception: {e}")


@dataclass
class TestCase:
    name: str
    input_data: Any
    expected: str | None = None
    max_duration_ms: float = 0


class AgentTestSuite:
    def __init__(self, name: str):
        self.name = name
        self.test_cases: list[TestCase] = []

    def add_case(self, name: str, input_data: Any, expected: str | None = None, max_duration_ms: float = 0):
        self.test_cases.append(TestCase(name=name, input_data=input_data, expected=expected, max_duration_ms=max_duration_ms))

    def run(self, agent) -> dict:
        results = []
        passed = 0
        failed = 0
        for case in self.test_cases:
            start = __import__("time").time()
            result = agent.run(case.input_data)
            duration = (__import__("time").time() - start) * 1000
            case_passed = True
            reason = ""
            if case.expected and case.expected not in str(result):
                case_passed = False
                reason = f"Expected '{case.expected}' in result"
            if case.max_duration_ms > 0 and duration > case.max_duration_ms:
                case_passed = False
                reason = f"Duration {duration:.0f}ms exceeded {case.max_duration_ms}ms"
            if case_passed:
                passed += 1
            else:
                failed += 1
            results.append({"name": case.name, "passed": case_passed, "reason": reason, "duration_ms": duration})
        return {"suite": self.name, "total": len(self.test_cases), "passed": passed, "failed": failed, "results": results}

    def print_report(self, results: dict):
        print(f"\n=== Test Suite: {results['suite']} ===")
        print(f"Total: {results['total']} | Passed: {results['passed']} | Failed: {results['failed']}")
        for r in results["results"]:
            icon = "✅" if r["passed"] else "❌"
            print(f"  {icon} {r['name']} ({r['duration_ms']:.0f}ms)")
            if r["reason"]:
                print(f"     {r['reason']}")