"""Agent 编排器 — 真·多 Agent 协作系统

每个 Agent 是独立实例：有自己的 memory、tools、state、identity。
通过 MessageBus 通信：委派(delegate)、直接(direct)、广播(broadcast)。

架构:
    User Task → Master.decompose()
                    ↓
              ┌─ Worker₁.act() ─┐
              ├─ Worker₂.act() ─┤  ← 并行执行（独立 context）
              └─ Worker₃.act() ─┘
                    ↓
              Master.aggregate()
                    ↓
              Final Result
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.state import AgentContext
from agent.memory import AgentMemory, get_memory
from agent.tools.registry import ToolRegistry
from agent.message_bus import MessageBus, MsgType
from agent.crew_agent import CrewAgent, AgentIdentity, TaskResult, ROLE_PROFILES
from agent.attn_router import AttnResRouter, AgentSignal

logger = logging.getLogger(__name__)


@dataclass
class CrewResult:
    """Crew 执行结果"""
    task: str
    subtask_count: int
    agent_results: list[TaskResult]
    final: str
    total_latency_ms: float
    consensus: dict[str, int] = field(default_factory=dict)  # 投票结果


class Crew:
    """Agent 编队管理"""

    def __init__(self, llm_client, model: str = "deepseek-chat", bus: Optional[MessageBus] = None):
        self.client = llm_client
        self.model = model
        self.bus = bus or MessageBus()
        self.agents: dict[str, CrewAgent] = {}

    def add(self, name: str, role: str, tools: Optional[ToolRegistry] = None, memory: Optional[AgentMemory] = None):
        """向 Crew 添加一个 Agent"""
        agent = CrewAgent(
            identity=AgentIdentity(
                name=name,
                role=role,
                description=ROLE_PROFILES.get(role, {}).get("description", ""),
                expertise=ROLE_PROFILES.get(role, {}).get("expertise", []),
            ),
            llm_client=self.client,
            model=self.model,
            memory=memory or get_memory(),
            tools=tools or ToolRegistry(),
            bus=self.bus,
        )
        self.bus.register(name)
        self.agents[name] = agent
        return agent

    def get(self, name: str) -> Optional[CrewAgent]:
        return self.agents.get(name)

    def list_roles(self) -> list[str]:
        return [a.identity.role for a in self.agents.values()]

    def stats(self) -> list[dict]:
        return [a.stats() for a in self.agents.values()]


class AgentOrchestrator:
    """多 Agent 编排器 — 真协作模式 + AttnRes选择性路由"""

    def __init__(self, llm_client, model: str = "deepseek-chat"):
        self.client = llm_client
        self.model = model
        self.router = AttnResRouter()

    async def execute_with_crew(self, task: str, crew: Crew) -> CrewResult:
        """使用真实 Agent Crew 执行复杂任务"""
        start = time.time()

        # 1. Master 拆解
        master = self._get_or_create_master(crew)
        subtasks = await self._decompose_via_agent(master, task)
        logger.info("Master 拆解出 %d 个子任务", len(subtasks))

        # 2. 按角色分配
        assignments = self._assign_subtasks(subtasks, crew)

        # 3. 拓扑排序后并行执行——每个子任务发给独立 Agent
        batches = self._topological_sort(subtasks)
        all_results: list[TaskResult] = []

        for batch in batches:
            batch_tasks = []
            for st in batch:
                agent = assignments.get(st.get("id"))
                if not agent:
                    agent = master  # fallback to master
                # 构造上下文：前序结果摘要
                context = self._build_context(all_results)
                batch_tasks.append(agent.act(st.get("description"), context))
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)

        # 4. Master 聚合 + 投票
        final = await self._aggregate_via_agent(master, task, all_results)
        consensus = self._vote(all_results)

        total_latency = (time.time() - start) * 1000
        logger.info("Crew 完成: %d 子任务, %.0fms", len(subtasks), total_latency)

        return CrewResult(
            task=task,
            subtask_count=len(subtasks),
            agent_results=all_results,
            final=final,
            total_latency_ms=total_latency,
            consensus=consensus,
        )

    # ── 拆解 ──

    async def _decompose_via_agent(self, master: CrewAgent, task: str) -> list[dict]:
        """让 Master Agent 拆解任务"""
        prompt = (
            f"Decompose this task into 2-5 subtasks. Each subtask must have an id, "
            f"a description, a role, and dependencies.\n\n"
            f"Roles available: planner, developer, reviewer, tester, researcher\n\n"
            f"Task: {task}\n\n"
            f"Output ONLY a JSON array:\n"
            f'[{{"id":"1","description":"...","role":"developer","depends_on":[]}}]\n\n'
            f"Prefer parallelism. Only add dependencies when truly necessary."
        )

        response = await master.think(prompt)
        try:
            text = response["content"].strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, KeyError):
            logger.warning("任务拆解 JSON 解析失败，使用单任务回退")
            return [{"id": "1", "description": task, "role": "developer", "depends_on": []}]

    # ── 分配 ──

    def _assign_subtasks(self, subtasks: list[dict], crew: Crew) -> dict[str, CrewAgent]:
        """将子任务分配给最匹配的 Agent"""
        assignments: dict[str, CrewAgent] = {}
        role_agents: dict[str, CrewAgent] = {}
        for agent in crew.agents.values():
            role_agents[agent.identity.role] = agent

        for st in subtasks:
            role = st.get("role", "developer")
            agent = role_agents.get(role)
            if not agent:
                # 回退：任何可用 Agent
                agent = list(crew.agents.values())[0] if crew.agents else None
            if agent:
                assignments[st["id"]] = agent

        return assignments

    # ── 拓扑排序 ──

    @staticmethod
    def _topological_sort(subtasks: list[dict]) -> list[list[dict]]:
        """按依赖关系分层，每层可并行"""
        completed: set[str] = set()
        batches: list[list[dict]] = []
        remaining = list(subtasks)

        while remaining:
            ready = [
                st for st in remaining
                if all(dep in completed for dep in st.get("depends_on", []))
            ]
            if not ready:
                batches.append(remaining)
                break
            batches.append(ready)
            completed.update(st["id"] for st in ready)
            remaining = [st for st in remaining if st not in ready]

        return batches

    # ── 上下文构造 ──

    @staticmethod
    def _build_context(prior_results: list[TaskResult]) -> str:
        """从前序结果构造上下文传给下游 Agent"""
        if not prior_results:
            return ""
        parts = ["What other agents have done so far:"]
        for r in prior_results:
            status = "OK" if r.success else "FAILED"
            parts.append(f"[{r.agent_name}/{status}] {r.content[:300]}")
        return "\n".join(parts)

    # ── 聚合 ──

    async def _aggregate_via_agent(self, master: CrewAgent, task: str, results: list[TaskResult]) -> str:
        """Master 聚合所有结果"""
        results_text = "\n---\n".join(
            f"[{r.agent_name}]({r.task_id}) success={r.success}\n{r.content}"
            for r in results
        )

        prompt = (
            f"Synthesize the following sub-agent results into a final, coherent answer.\n\n"
            f"Original task: {task}\n\n"
            f"Sub-results:\n{results_text}\n\n"
            f"Provide the final synthesized output."
        )

        final = await master.think(prompt)
        return final["content"]

    # ── AttnRes 路由增强 ──

    async def execute_with_routing(
        self, task: str, crew: Crew, mode: str = "full"
    ) -> dict:
        """Execute with AttnRes selective routing over agent outputs."""
        import time as _time
        start = _time.time()

        result = await self.execute_with_crew(task, crew)

        signals = []
        for r in result.agent_results:
            role = getattr(r, "role", "unknown")
            agent_id = getattr(r, "agent_id", role)
            content = r.result if r.result else (r.error if r.error else "")
            signals.append(AgentSignal(
                agent_id=agent_id, role=role, content=content[:2000],
                confidence=1.0 if r.success else 0.0,
                metadata={"success": r.success, "latency_ms": r.latency_ms},
            ))

        routing = self.router.compare(task, signals)
        best = self.router.route(task, signals, mode=mode)

        return {
            "task": task,
            "crew_result": result,
            "routing": {
                "mode": mode,
                "comparison": routing,
                "weights": {k: round(v, 3) for k, v in best.weights.items()},
                "confidence": round(best.confidence, 3),
                "top_agent": max(best.weights, key=best.weights.get) if best.weights else None,
                "aggregated": best.aggregated[:2000],
            },
            "latency_ms": (_time.time() - start) * 1000,
        }

    # ── 投票 ──

    @staticmethod
    def _vote(results: list[TaskResult]) -> dict[str, int]:
        """从结果中做简单多数投票（基于成功/失败和关键词）"""
        votes: dict[str, int] = {}
        for r in results:
            if r.success:
                key = "success"
            elif r.error:
                key = f"error:{r.error[:50]}"
            else:
                key = "unknown"
            votes[key] = votes.get(key, 0) + 1
        return votes

    # ── 辅助 ──

    def _get_or_create_master(self, crew: Crew) -> CrewAgent:
        """获取或创建 Master Agent"""
        master = crew.get("master")
        if not master:
            master = crew.add("master", "master")
        return master


# ── 工厂函数 ──

def create_crew(client, model: str = "deepseek-chat", roles: Optional[list[str]] = None) -> Crew:
    """创建默认 Crew

    Args:
        client: LLM client
        model: model name
        roles: agent roles to include (default: developer, reviewer, tester)
    """
    roles = roles or ["developer", "reviewer", "tester"]
    crew = Crew(client, model)
    for i, role in enumerate(roles):
        crew.add(f"{role}-{i+1}", role)
    # Always add master
    crew.add("master", "master")
    return crew
