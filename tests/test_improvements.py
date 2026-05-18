"""测试从 Hermes Agent / Anthropic Skills / Vercel AI SDK 学习的改进"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 技能系统测试 ──

class TestSkills:
    def test_parse_frontmatter_standard(self):
        from agent.skills import parse_frontmatter

        content = """---
name: my-skill
description: A test skill
tags: [python, agent]
---
# Instructions

Do this, then that."""

        meta, body = parse_frontmatter(content)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A test skill"
        assert "Do this" in body

    def test_parse_frontmatter_simple(self):
        from agent.skills import parse_frontmatter

        content = """---
name: simple
---
Just do X."""

        meta, body = parse_frontmatter(content)
        assert meta["name"] == "simple"
        assert "Just do X" in body

    def test_parse_no_frontmatter(self):
        from agent.skills import parse_frontmatter

        content = "# Just a doc\n\nNo frontmatter here."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_build_frontmatter(self):
        from agent.skills import build_frontmatter

        fm = build_frontmatter("test-skill", "For testing", source="auto")
        assert "name: test-skill" in fm
        assert "description: For testing" in fm
        assert fm.startswith("---")
        assert fm.endswith("---")

    def test_skill_manager_create_and_search(self):
        from agent.skills import SkillManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = SkillManager(Path(tmp))

            # 创建
            mgr.create("fastapi-setup", "Set up FastAPI projects", "## Steps\n1. Install\n2. Configure")
            mgr.create("docker-deploy", "Docker deployment guide", "## Steps\n1. Build\n2. Run")

            # 搜索
            results = mgr.search("fastapi")
            assert len(results) >= 1
            assert results[0].name == "fastapi-setup"

            # 无匹配
            results = mgr.search("nonexistent")
            assert len(results) == 0

            # 列表
            all_skills = mgr.list_all()
            assert len(all_skills) >= 2

            # 获取
            skill = mgr.get("fastapi-setup")
            assert skill is not None
            assert "Install" in skill.body

    def test_skill_manager_get_nonexistent(self):
        from agent.skills import SkillManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SkillManager(Path(tmp))
            assert mgr.get("nonexistent") is None

    def test_inject_context(self):
        from agent.skills import SkillManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = SkillManager(Path(tmp))
            mgr.create("python-testing", "Write Python tests with pytest", "## Use pytest")
            mgr.create("fastapi-routes", "FastAPI route patterns", "## Use decorators")

            ctx = mgr.inject_context("I need to test my FastAPI app")
            assert "python-testing" in ctx or "fastapi-routes" in ctx


# ── AST 工具发现测试 ──

class TestASTDiscovery:
    def test_has_tool_pattern_true(self):
        from agent.tools import _has_tool_pattern
        import agent.tools as pkg

        pkg_path = Path(pkg.__path__[0])
        calc = pkg_path / "calc_tool.py"
        assert _has_tool_pattern(str(calc))

    def test_has_tool_pattern_false(self):
        from agent.tools import _has_tool_pattern
        import agent.tools as pkg

        pkg_path = Path(pkg.__path__[0])
        registry_file = pkg_path / "registry.py"
        assert not _has_tool_pattern(str(registry_file))

    def test_has_tool_pattern_syntax_error(self):
        from agent.tools import _has_tool_pattern

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("this is not valid ) python @@@")
            path = f.name
        try:
            assert not _has_tool_pattern(path)
        finally:
            os.unlink(path)

    def test_register_all_runs(self):
        from agent.tools.registry import ToolRegistry
        from agent.tools import register_all

        registry = ToolRegistry()
        register_all(registry)

        tools = list(registry._tools.keys())
        assert len(tools) > 0


# ── 上下文压缩测试 ──

class MockCompressorClient:
    class chat:
        class completions:
            @staticmethod
            async def create(*, model, messages, max_tokens=300, temperature=0.1):
                class Fake:
                    class Choice:
                        class Msg:
                            content = "Summary: user asked about X, agent used tools Y and Z."
                        message = Msg()
                    choices = [Choice()]
                return Fake()


class TestContextCompressor:
    def test_truncate_small(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(MockCompressorClient())
        msgs = [{"role": "user", "content": "hi"}] * 3
        result = comp._truncate(msgs, max_recent=4)
        assert len(result) == 3  # too small, no truncation needed

    def test_truncate_large(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(MockCompressorClient())
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "q4"},
        ]
        result = comp._truncate(msgs, max_recent=3)
        assert "q1" not in str(result)  # 旧消息被截断
        assert "q4" in str(result)      # 最新消息保留
        assert result[0]["role"] == "system"  # 系统消息保留

    def test_compress_hybrid(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(MockCompressorClient())
        msgs = [
            {"role": "user", "content": f"msg{i}"} for i in range(10)
        ]

        async def run():
            result = await comp.compress(msgs, max_recent=3, mode="hybrid")
            # 压缩后应少于原始10条
            assert len(result) < 10
            # 最后3条完整消息保留
            assert "msg9" in str(result)
            assert "msg8" in str(result)

        asyncio.run(run())

    def test_compress_truncate_no_client(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(MockCompressorClient())
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]

        result = comp._truncate(msgs, max_recent=3)
        assert len(result) == 3
        assert "m7" in str(result[-1])


# ── 技能 + 压缩联动 ──

class TestIntegration:
    def test_skill_injection_into_context(self):
        from agent.skills import SkillManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = SkillManager(Path(tmp))
            mgr.create("api-design", "RESTful API design patterns", "## Use proper status codes")

            ctx = mgr.inject_context("design a user registration API")
            assert len(ctx) > 0

    def test_compress_then_skill_context(self):
        from agent.skills import SkillManager
        from agent.context_compressor import ContextCompressor

        with tempfile.TemporaryDirectory() as tmp:
            mgr = SkillManager(Path(tmp))
            mgr.create("code-review", "Code review checklist", "## Check security, style, tests")
            mgr.create("testing-guide", "Testing best practices", "## Write tests first")

            ctx = mgr.inject_context("review my FastAPI code and write tests")
            assert "code-review" in ctx or "testing-guide" in ctx

            comp = ContextCompressor(MockCompressorClient())
            msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
            result = comp._truncate(msgs, max_recent=3)
            assert len(result) == 3


if __name__ == "__main__":
    import pytest as pt
    sys.exit(pt.main([__file__, "-v", "--tb=short"]))
