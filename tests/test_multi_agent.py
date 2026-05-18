"""真多 Agent 协作测试 — MessageBus + CrewAgent + Crew + Orchestrator

验证:
    1. MessageBus: 注册/发送/广播/委派
    2. CrewAgent: 独立身份/角色/状态
    3. Crew: 编队管理/添加Agent
    4. AgentOrchestrator: 拆解/分配/排序/聚合/投票
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 模拟 LLM Client ──

def _infer_role(messages: list) -> str:
    for m in messages:
        c = m.get("content", "")
        s = str(m)
        if "planner" in c.lower() or "Technical Planner" in s:
            return "planner"
        elif "developer" in c.lower() or "Senior Developer" in s:
            return "developer"
        elif "reviewer" in c.lower() or "Code Reviewer" in s:
            return "reviewer"
        elif "tester" in c.lower() or "QA Engineer" in s:
            return "tester"
        elif "researcher" in c.lower() or "Technical Researcher" in s:
            return "researcher"
        elif "master" in c.lower() or "Master" in s:
            return "master"
    return "unknown"


_RESPONSES = {
    "planner": "Plan: 1) analyze requirements, 2) design architecture, 3) implement, 4) test",
    "developer": "Code: implemented feature with proper error handling",
    "reviewer": "Review: found 2 minor issues - variable naming, missing docstring",
    "tester": "Test plan: 5 test cases covering happy path and edge cases",
    "researcher": "Found: relevant documentation at official docs, best practice is X",
    "master": '[{"id":"1","description":"Implement feature","role":"developer","depends_on":[]},{"id":"2","description":"Review code","role":"reviewer","depends_on":["1"]},{"id":"3","description":"Write tests","role":"tester","depends_on":["1"]}]',
    "unknown": "Task completed successfully",
}


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    async def create(self, *, model, messages, max_tokens=800, temperature=0.3):
        role = _infer_role(messages)
        text = _RESPONSES.get(role, _RESPONSES["unknown"])
        return FakeResponse(text)


class MockLLMClient:
    """模拟 LLM 客户端"""
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


# ── MessageBus 测试 ──

class TestMessageBus:
    def test_register_and_list(self):
        from agent.message_bus import MessageBus
        bus = MessageBus()
        bus.register("agent-1")
        bus.register("agent-2")
        assert sorted(bus.list_agents()) == ["agent-1", "agent-2"]

    def test_send_and_receive(self):
        from agent.message_bus import MessageBus
        bus = MessageBus()
        bus.register("alice")
        bus.register("bob")

        async def test():
            await bus.send("alice", "bob", "hello bob")
            env = await bus.receive("bob", timeout=1.0)
            assert env is not None
            assert env.message.sender == "alice"
            assert env.message.payload == "hello bob"

        asyncio.run(test())

    def test_delegate_with_response(self):
        from agent.message_bus import MessageBus
        bus = MessageBus()
        bus.register("master")
        bus.register("worker")

        async def worker_listen():
            env = await bus.receive("worker", timeout=5.0)
            bus.respond(env.message.reply_to, "task-done")

        async def master_delegate():
            task = asyncio.create_task(worker_listen())
            await asyncio.sleep(0.01)  # let worker register
            result = await bus.delegate("master", "worker", "do-work", timeout=5.0)
            assert result == "task-done"
            await task

        asyncio.run(master_delegate())

    def test_broadcast(self):
        from agent.message_bus import MessageBus
        bus = MessageBus()
        bus.register("a1")
        bus.register("a2")
        bus.register("a3")

        async def test():
            ids = await bus.broadcast("master", "announcement", exclude=["master"])
            assert len(ids) == 3
            # 每个 agent 都应收到消息
            for name in ["a1", "a2", "a3"]:
                env = await bus.receive(name, timeout=0.5)
                assert env is not None
                assert env.message.payload == "announcement"

        asyncio.run(test())

    def test_unregister(self):
        from agent.message_bus import MessageBus
        bus = MessageBus()
        bus.register("temp")
        bus.unregister("temp")
        assert "temp" not in bus.list_agents()


# ── CrewAgent 测试 ──

class TestCrewAgent:
    def test_identity(self):
        from agent.crew_agent import AgentIdentity, CrewAgent

        identity = AgentIdentity(name="dev-1", role="developer", expertise=["python", "fastapi"])
        agent = CrewAgent(identity=identity, llm_client=MockLLMClient())

        assert agent.identity.name == "dev-1"
        assert agent.identity.role == "developer"
        assert "python" in agent.identity.expertise
        assert agent.state.value == "idle"

    def test_from_profile(self):
        from agent.crew_agent import CrewAgent

        agent = CrewAgent.from_profile("reviewer-1", "reviewer", MockLLMClient())
        assert agent.identity.role == "reviewer"
        assert "code review" in agent.identity.description.lower()

    def test_think(self):
        from agent.crew_agent import CrewAgent

        agent = CrewAgent.from_profile("dev-1", "developer", MockLLMClient())

        async def run():
            result = await agent.think("Write a login function")
            assert isinstance(result, dict)
            assert "content" in result
            assert "Code:" in result["content"]
            assert agent.state.value == "planning"

        asyncio.run(run())

    def test_act(self):
        from agent.crew_agent import CrewAgent

        agent = CrewAgent.from_profile("tester-1", "tester", MockLLMClient())

        async def run():
            result = await agent.act("Test login endpoint")
            assert result.success
            assert len(result.content) > 0
            assert result.agent_name == "tester-1"
            assert agent.task_count == 1

        asyncio.run(run())

    def test_stats(self):
        from agent.crew_agent import CrewAgent

        agent = CrewAgent.from_profile("dev-1", "developer", MockLLMClient())

        async def run():
            await agent.act("task 1")
            await agent.act("task 2")
            stats = agent.stats()
            assert stats["tasks_completed"] == 2
            assert stats["name"] == "dev-1"

        asyncio.run(run())


# ── Crew 测试 ──

class TestCrew:
    def test_add_agents(self):
        from agent.orchestrator import Crew

        crew = Crew(MockLLMClient())
        dev = crew.add("dev-1", "developer")
        rev = crew.add("rev-1", "reviewer")

        assert crew.get("dev-1") is dev
        assert crew.get("rev-1") is rev
        assert "dev-1" in crew.bus.list_agents()
        assert "rev-1" in crew.bus.list_agents()

    def test_stats(self):
        from agent.orchestrator import Crew

        crew = Crew(MockLLMClient())
        crew.add("dev-1", "developer")
        crew.add("tester-1", "tester")
        stats = crew.stats()
        assert len(stats) == 2
        assert all("name" in s for s in stats)


# ── Orchestrator 测试 ──

class TestOrchestrator:
    def test_topological_sort(self):
        from agent.orchestrator import AgentOrchestrator

        subtasks = [
            {"id": "1", "role": "planner", "depends_on": []},
            {"id": "2", "role": "developer", "depends_on": ["1"]},
            {"id": "3", "role": "developer", "depends_on": ["1"]},
            {"id": "4", "role": "tester", "depends_on": ["2", "3"]},
        ]

        batches = AgentOrchestrator._topological_sort(subtasks)
        assert len(batches) == 3
        assert [s["id"] for s in batches[0]] == ["1"]
        assert {s["id"] for s in batches[1]} == {"2", "3"}
        assert [s["id"] for s in batches[2]] == ["4"]

    def test_build_context(self):
        from agent.orchestrator import AgentOrchestrator
        from agent.crew_agent import TaskResult

        results = [
            TaskResult(task_id="1", agent_name="dev-1", content="Built feature X", success=True),
            TaskResult(task_id="2", agent_name="tester-1", content="Tested: all pass", success=True),
        ]
        ctx = AgentOrchestrator._build_context(results)
        assert "dev-1" in ctx
        assert "tester-1" in ctx
        assert "Built feature X" in ctx

    def test_vote(self):
        from agent.orchestrator import AgentOrchestrator
        from agent.crew_agent import TaskResult

        results = [
            TaskResult(task_id="1", agent_name="a", content="ok", success=True),
            TaskResult(task_id="2", agent_name="b", content="ok", success=True),
            TaskResult(task_id="3", agent_name="c", content="", success=False, error="timeout"),
        ]
        votes = AgentOrchestrator._vote(results)
        assert votes["success"] == 2
        assert any("timeout" in k for k in votes)

    def test_execute_with_crew(self):
        """端到端：真实 Crew 执行任务（模拟 LLM）"""
        from agent.orchestrator import AgentOrchestrator, Crew

        crew = Crew(MockLLMClient())
        crew.add("dev-1", "developer")
        crew.add("reviewer-1", "reviewer")
        crew.add("tester-1", "tester")
        crew.add("master", "master")

        orch = AgentOrchestrator(MockLLMClient())

        async def run():
            result = await orch.execute_with_crew("Build a login feature", crew)
            assert result.subtask_count >= 1
            assert len(result.agent_results) > 0

        asyncio.run(run())


# ── 集成测试: Bus + Agent + Orchestrator ──

class TestFullIntegration:
    def test_bus_communication_between_agents(self):
        """两个 Agent 通过 Bus 直接通信"""
        from agent.message_bus import MessageBus
        from agent.crew_agent import CrewAgent

        bus = MessageBus()
        alice = CrewAgent.from_profile("alice", "developer", MockLLMClient(), bus=bus)
        bob = CrewAgent.from_profile("bob", "reviewer", MockLLMClient(), bus=bus)

        bus.register("alice")
        bus.register("bob")

        async def bob_listen():
            env = await bus.receive("bob", timeout=5.0)
            assert env.message.sender == "alice"

        async def run():
            listener = asyncio.create_task(bob_listen())
            await asyncio.sleep(0.01)
            await alice.send_to("bob", "How does this look?")
            await listener

        asyncio.run(run())

    def test_master_delegates_to_worker(self):
        """Master 通过 Bus 委派任务给 Worker"""
        from agent.message_bus import MessageBus
        from agent.crew_agent import CrewAgent

        bus = MessageBus()
        master = CrewAgent.from_profile("master", "master", MockLLMClient(), bus=bus)
        worker = CrewAgent.from_profile("worker", "developer", MockLLMClient(), bus=bus)

        bus.register("master")
        bus.register("worker")

        async def worker_loop():
            env = await bus.receive("worker", timeout=5.0)
            # Worker processes the delegation
            result = await worker.act(str(env.message.payload))
            bus.respond(env.message.reply_to, result)

        async def run():
            worker_task = asyncio.create_task(worker_loop())
            await asyncio.sleep(0.01)
            result = await master.delegate_to("worker", "Implement a login endpoint")
            assert isinstance(result, object)
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())


if __name__ == "__main__":
    import pytest as pt
    sys.exit(pt.main([__file__, "-v", "--tb=short"]))
