"""SuperAgent integration tests — full loops with mock LLM clients.

Tests the three SuperAgent engines end-to-end:
  1. Reflect→Action: tool degradation
  2. Multi-Model Debate: 4-round arbitration
  3. Skills Bootstrapping: code generation + validation + registration
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.state import AgentContext
from agent.tools.registry import ToolRegistry
from agent.governance import GovernancePanel
from agent.sandbox import SandboxExecutor
from agent.reflect_action import ReflectActionEngine
from agent.bootstrap import BootstrapEngine, BootstrappedTool


# ── Mock Helpers ──────────────────────────────────

def make_mock_client(responses: list[str]):
    """Create an AsyncOpenAI mock that returns a sequence of responses."""
    client = MagicMock()
    call_count = [0]

    async def create(*args, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        resp_text = responses[idx]
        call_count[0] += 1

        choice = MagicMock()
        choice.message.content = resp_text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client.chat.completions.create = AsyncMock(side_effect=create)
    return client


def make_stream_client(chunks: list[list[dict]]):
    """Create a streaming client that yields chunks across multiple calls."""
    client = MagicMock()
    call_index = [0]

    def make_async_gen(chunk_list):
        async def gen():
            for c in chunk_list:
                yield type('Chunk', (), {
                    'content': c.get("content"),
                    'tool_calls': c.get("tool_calls", []),
                })()
        return gen()

    # We'll set up the mock differently since streaming is async
    return client


# ── Reflect→Action Tests ──────────────────────────

class TestReflectActionEngine:
    """Test that tool failure triggers degradation and alternatives."""

    def test_no_degradation_on_success(self):
        engine = ReflectActionEngine(failure_threshold=3)
        for _ in range(5):
            engine.record_tool_result("web_search", success=True)
        assert not engine.is_degraded("web_search")
        assert engine.status()["degraded_tools"] == []

    def test_degradation_after_threshold(self):
        engine = ReflectActionEngine(failure_threshold=3)
        for i in range(3):
            engine.record_tool_result("web_search", success=False, error=f"timeout {i}")
        assert engine.is_degraded("web_search")
        assert "web_search" in engine.status()["degraded_tools"]

    def test_alternatives_provided(self):
        engine = ReflectActionEngine(failure_threshold=3)
        for _ in range(3):
            engine.record_tool_result("web_search", success=False, error="timeout")
        alts = engine.get_alternatives("web_search")
        assert len(alts) > 0
        assert any(a in alts for a in ["web_fetch"])

    def test_success_resets_failure_count(self):
        engine = ReflectActionEngine(failure_threshold=3)
        engine.record_tool_result("web_search", success=False, error="fail1")
        engine.record_tool_result("web_search", success=False, error="fail2")
        engine.record_tool_result("web_search", success=True)
        engine.record_tool_result("web_search", success=False, error="fail3")
        assert not engine.is_degraded("web_search")

    def test_filter_degraded_tools_substitutes(self):
        engine = ReflectActionEngine(failure_threshold=2)
        for _ in range(2):
            engine.record_tool_result("web_search", success=False)
        tool_calls = [
            {"function": {"name": "web_search", "arguments": '{"q": "test"}'}},
            {"function": {"name": "read_file", "arguments": '{"path": "test.txt"}'}},
        ]
        filtered = engine.filter_degraded_tools(tool_calls)
        assert len(filtered) == 2
        # web_search should be substituted
        names = [tc["function"]["name"] for tc in filtered]
        assert "web_search" not in names or len(engine.get_alternatives("web_search")) == 0
        assert "read_file" in names

    def test_evaluate_detects_repeated_loop(self):
        engine = ReflectActionEngine(failure_threshold=3)
        tool_results = [
            {"tool": "web_search", "status": "error", "result": "timeout"},
            {"tool": "web_search", "status": "error", "result": "timeout"},
            {"tool": "web_search", "status": "error", "result": "timeout"},
        ]
        actions = engine.evaluate("web_search keeps failing", tool_results)
        assert any(a["type"] == "pivot" for a in actions)

    def test_evaluate_detects_missing_tool(self):
        engine = ReflectActionEngine(failure_threshold=3)
        reflection = "I need a tool to parse HTML tables but there is no tool available for this"
        actions = engine.evaluate(reflection, [])
        assert any(a["type"] == "missing_tool" for a in actions)


# ── Bootstrap Tests ───────────────────────────────

class TestBootstrap:
    """Test that code generation + validation works end-to-end."""

    def test_syntax_validation_accepts_valid_code(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = '''def parse_table(params: dict) -> str:
    """Parse a markdown table and return JSON."""
    rows = params.get("markdown", "").split("\\n")
    return str(len(rows))
'''
        assert engine._validate_syntax(code, "parse_table")

    def test_syntax_validation_rejects_invalid(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = "def broken(:"
        assert not engine._validate_syntax(code, "broken")

    def test_validation_blocks_os_import(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = '''def dangerous(params: dict) -> str:
    import os
    os.system("rm -rf /")
    return "done"
'''
        assert not engine._validate_syntax(code, "dangerous")

    def test_validation_blocks_subprocess_import(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = '''def dangerous(params: dict) -> str:
    import subprocess
    subprocess.run("evil.exe")
    return "done"
'''
        assert not engine._validate_syntax(code, "dangerous")

    def test_validation_blocks_socket_import(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = '''def dangerous(params: dict) -> str:
    import socket
    return "done"
'''
        assert not engine._validate_syntax(code, "dangerous")

    def test_validation_requires_function_signature(self):
        engine = BootstrapEngine.__new__(BootstrapEngine)
        code = "x = 1"
        assert not engine._validate_syntax(code, "not_a_function")

    def test_register_tool_in_registry(self):
        """Generated tool code registered and callable."""
        registry = ToolRegistry()
        engine = BootstrapEngine.__new__(BootstrapEngine)
        engine._tools = {}
        code = '''def hello_world(params: dict) -> str:
    """Say hello."""
    name = params.get("name", "world")
    return f"Hello, {name}!"
'''
        tool = BootstrappedTool(
            name="hello_world",
            code=code,
            description="Says hello",
            created_at="2026-05-18T00:00:00Z",
            source_reflection="need a greeting tool",
            validated=True,
        )
        success = engine.register_tool(tool, registry)
        assert success
        assert tool.registered

        # Verify it's callable through registry
        result = registry.execute("hello_world", {"name": "Test"})
        assert "Hello, Test!" in result

    def test_generate_from_reflection(self):
        """Full generation with mock LLM, validate, register."""
        client = make_mock_client([
            '''def count_words(params: dict) -> str:
    """Count words in text."""
    text = params.get("text", "")
    words = text.split()
    return str(len(words))
'''
        ])
        engine = BootstrapEngine(client, model="test-model")
        registry = ToolRegistry()

        tool = asyncio.run(engine.generate_from_reflection(
            "I need a tool to count words but don't have one",
            "count_words",
        ))
        assert tool.validated
        assert "count_words" in tool.code or "def " in tool.code

        success = engine.register_tool(tool, registry)
        assert success

        result = registry.execute("count_words", {"text": "one two three"})
        assert "3" in result


# ── Debate Tests ──────────────────────────────────

class TestDebateEngine:
    """Test the debate engine with mock model clients."""

    def test_debate_returns_consensus(self):
        from agent.debate import DebateEngine

        primary = make_mock_client(["The answer is 42."])
        challenger = make_mock_client(["The answer should be verified."])
        arbiter = make_mock_client(["After review: the answer is 42."])

        engine = DebateEngine(primary, challenger, arbiter)

        result = asyncio.run(engine.debate(
            task="What is the answer?",
            primary_model="deepseek-chat",
            challenger_model="qwen2.5:7b",
        ))
        assert result.completed
        assert result.total_rounds == 4
        assert len(result.rounds) == 4
        assert result.rounds[0].speaker == "primary"
        assert result.rounds[1].speaker == "challenger"
        assert result.rounds[2].speaker == "primary_rebuttal"
        assert result.rounds[3].speaker == "arbitrator"
        assert "42" in result.consensus

    def test_debate_stores_results(self):
        from agent.debate import DebateEngine

        primary = make_mock_client(["Proposal A"])
        challenger = make_mock_client(["Critique of Proposal A"])
        arbiter = make_mock_client(["Consensus: A with modifications"])

        engine = DebateEngine(primary, challenger, arbiter)

        result = asyncio.run(engine.debate(
            task="Solve this problem",
            primary_model="gpt-4",
            challenger_model="claude-3",
        ))
        assert result.debate_id.startswith("debate-")

        retrieved = engine.get_result(result.debate_id)
        assert retrieved is not None
        assert retrieved.completed

    def test_debate_status(self):
        from agent.debate import DebateEngine

        primary = make_mock_client(["Answer."])
        challenger = make_mock_client(["Critique."])
        arbiter = make_mock_client(["Consensus."])

        engine = DebateEngine(primary, challenger, arbiter)
        status = engine.status()
        assert "completed_debates" in status
        assert status["completed_debates"] == 0

        asyncio.run(engine.debate(task="Task", primary_model="m1", challenger_model="m2"))
        status = engine.status()
        assert status["completed_debates"] == 1

    def test_debate_error_handling(self):
        from agent.debate import DebateEngine

        # Challenger fails on first call
        async def always_fail(*args, **kwargs):
            raise RuntimeError("Model unavailable")

        bad_client = MagicMock()
        bad_client.chat.completions.create = AsyncMock(side_effect=always_fail)
        primary = make_mock_client(["ok"])
        arbiter = make_mock_client(["arbitration"])

        engine = DebateEngine(primary, bad_client, arbiter)
        result = asyncio.run(engine.debate(
            task="Test", primary_model="m1", challenger_model="m2"
        ))
        assert not result.completed
        assert result.error


# ── AsyncAgent SuperAgent Integration ──────────────

class TestAsyncAgentSuperModes:
    """Test AsyncAgent with SuperAgent features enabled."""

    def test_super_agent_initialization(self):
        from agent.async_core import AsyncAgent

        client = make_mock_client(["Hello"])
        registry = ToolRegistry()
        registry.register("echo", "echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]},
                         lambda text: text)

        agent = AsyncAgent(
            client=client,
            model="deepseek-chat",
            registry=registry,
            enable_super_agent=True,
            challenger_client=client,
            challenger_model="qwen2.5:7b",
        )
        assert agent.reflect_action is not None
        assert agent.bootstrap is not None
        assert agent.debate_engine is not None
        assert agent.enable_super_agent

    def test_degrade_tool_manual(self):
        from agent.async_core import AsyncAgent

        client = make_mock_client(["ok"])
        registry = ToolRegistry()
        agent = AsyncAgent(
            client=client,
            model="test",
            registry=registry,
            enable_super_agent=True,
        )
        result = agent.degrade_tool("web_search")
        assert "degraded" in result
        assert result["alternatives"]

    def test_super_status(self):
        from agent.async_core import AsyncAgent

        client = make_mock_client(["ok"])
        registry = ToolRegistry()
        agent = AsyncAgent(
            client=client,
            model="test",
            registry=registry,
            enable_super_agent=True,
        )
        status = agent.get_super_status()
        assert "reflect_action" in status
        assert "debate" in status
        assert "bootstrap" in status

    def test_agent_run_preserves_reflection_with_super(self):
        from agent.async_core import AsyncAgent

        client = make_mock_client(["I'll answer directly.", ""])
        registry = ToolRegistry()

        agent = AsyncAgent(
            client=client,
            model="test",
            registry=registry,
            enable_super_agent=True,
            enable_reflection=True,
            enable_learning=True,
        )
        ctx = AgentContext(trace_id="test_super_001", max_steps=3)
        ctx = asyncio.run(agent.run(ctx, "Hello"))
        assert ctx.state.value in ("done", "error")
        # Should have at least one message from assistant
        assert len(ctx.messages) > 0

    def test_debate_run_fallback_no_challenger(self):
        from agent.async_core import AsyncAgent

        client = make_mock_client(["Self-critiqued answer"])
        registry = ToolRegistry()

        agent = AsyncAgent(
            client=client,
            model="test",
            registry=registry,
            enable_super_agent=True,
        )
        result = asyncio.run(agent.debate_run("What is 2+2?"))
        assert result.completed
        assert "Self-critiqued answer" in result.consensus


# ── Full Pipeline: Reflect → Bootstrap ────────────

class TestReflectToBootstrapPipeline:
    """Test that reflection gap detection flows into bootstrapping."""

    def test_missing_tool_triggers_bootstrap_generation(self):
        """End-to-end: reflect detects gap → bootstrap generates code → registers."""
        client = make_mock_client([
            '''def extract_urls(params: dict) -> str:
    """Extract URLs from text."""
    import re
    text = params.get("text", "")
    urls = re.findall(r'https?://[^\\s]+', text)
    return str(urls)
'''
        ])
        engine = BootstrapEngine(client, model="test-model")
        registry = ToolRegistry()

        tool = asyncio.run(engine.generate_from_reflection(
            "I need a tool to extract URLs from text but don't have one",
            "extract_urls",
        ))
        assert tool.validated
        assert "extract_urls" in tool.code

        registered = engine.register_tool(tool, registry)
        assert registered

        result = registry.execute("extract_urls",
                                 {"text": "Visit https://example.com and http://test.org"})
        assert "example.com" in result
        assert "test.org" in result

    def test_bootstrap_list_tracks_generated_tools(self):
        client = make_mock_client([
            '''def tool_a(params: dict) -> str:
    return "a"
'''
        ])
        engine = BootstrapEngine(client, model="test-model")
        asyncio.run(engine.generate_from_reflection("need tool a", "tool_a"))
        tools = engine.list_bootstrapped()
        assert any(t["name"] == "tool_a" for t in tools)
