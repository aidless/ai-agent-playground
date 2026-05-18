"""CrewAgent — 独立的多 Agent 实例

每个 CrewAgent 拥有：
    - 独立身份（name + role + expertise）
    - 独立记忆（AgentMemory 实例）
    - 独立工具集（ToolRegistry 子集）
    - 独立状态（AgentContext）
    - 通过 MessageBus 与其他 Agent 通信

Agent 生命周期:
    CREATED → IDLE → (receive task) → THINKING → ACTING → DONE/ERROR → IDLE
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.state import AgentContext, AgentState
from agent.memory import AgentMemory, get_memory
from agent.tools.registry import ToolRegistry
from agent.message_bus import MessageBus, Envelope, MsgType

logger = logging.getLogger(__name__)

# ── Agent 角色定义 ──────────────────────────────

ROLE_PROFILES: dict[str, dict] = {
    "planner": {
        "description": "Technical planner and architect",
        "expertise": ["system design", "task decomposition", "architecture"],
        "tools": ["web_search", "read_file", "list_files"],
        "system_prompt": (
            "You are a Technical Planner. Break complex requirements into actionable steps. "
            "Identify dependencies, risks, and the most efficient execution order. "
            "Communicate your plans clearly to other agents."
        ),
    },
    "developer": {
        "description": "Senior software developer",
        "expertise": ["coding", "debugging", "implementation"],
        "tools": ["read_file", "write_file", "run_python", "web_search", "web_fetch"],
        "system_prompt": (
            "You are a Senior Developer. Write clean, working code. "
            "Follow best practices. Handle edge cases. "
            "When stuck, ask for clarification rather than guessing."
        ),
    },
    "reviewer": {
        "description": "Code reviewer and quality assurance",
        "expertise": ["code review", "bug detection", "security audit"],
        "tools": ["read_file", "web_search"],
        "system_prompt": (
            "You are a Code Reviewer. Find bugs, security issues, and anti-patterns. "
            "Suggest concrete improvements. Be thorough but constructive. "
            "Focus on correctness, security, and readability."
        ),
    },
    "tester": {
        "description": "Test engineer",
        "expertise": ["test design", "edge cases", "quality"],
        "tools": ["read_file", "write_file", "run_python"],
        "system_prompt": (
            "You are a QA Engineer. Design test cases that cover happy paths, "
            "edge cases, and error conditions. Think about what could go wrong. "
            "Write tests that are readable and maintainable."
        ),
    },
    "researcher": {
        "description": "Technical researcher",
        "expertise": ["research", "fact-finding", "documentation"],
        "tools": ["web_search", "web_fetch", "read_file"],
        "system_prompt": (
            "You are a Technical Researcher. Find relevant facts, documentation, "
            "and alternative approaches. Provide concise, sourced information. "
            "Distinguish between facts and opinions."
        ),
    },
    "master": {
        "description": "Master orchestrator — decomposes tasks and assigns to crew",
        "expertise": ["orchestration", "task decomposition", "result synthesis"],
        "tools": ["web_search", "read_file"],
        "system_prompt": (
            "You are a Master Orchestrator. Decompose complex tasks into subtasks, "
            "assign them to the right specialists, and synthesize the results. "
            "Think about dependencies, parallelism, and quality control."
        ),
    },
}


@dataclass
class AgentIdentity:
    """Agent 身份"""
    name: str          # 唯一名称
    role: str          # planner/developer/reviewer/tester/researcher/master
    description: str = ""
    expertise: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    agent_name: str
    content: str
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0
    tool_calls_made: int = 0


class CrewAgent:
    """独立的多 Agent 实例"""

    def __init__(
        self,
        identity: AgentIdentity,
        llm_client,
        model: str = "deepseek-chat",
        memory: Optional[AgentMemory] = None,
        tools: Optional[ToolRegistry] = None,
        bus: Optional[MessageBus] = None,
    ):
        self.identity = identity
        self.client = llm_client
        self.model = model
        self.memory = memory or get_memory()
        self.tools = tools or ToolRegistry()
        self.bus = bus
        self.state = AgentState.IDLE
        self.task_count = 0
        self.total_latency_ms = 0.0

    @classmethod
    def from_profile(cls, name: str, role: str, llm_client, model: str = "deepseek-chat", bus: Optional[MessageBus] = None) -> "CrewAgent":
        """从角色模板创建 Agent"""
        profile = ROLE_PROFILES.get(role, ROLE_PROFILES["developer"])
        identity = AgentIdentity(
            name=name,
            role=role,
            description=profile["description"],
            expertise=profile.get("expertise", []),
        )
        return cls(identity=identity, llm_client=llm_client, model=model, bus=bus)

    # ── 核心方法 ──

    async def think(self, task: str, context: str = "", tool_choice: str = "auto") -> dict:
        """Agent 思考：返回 {"content": ..., "tool_calls": [...]}"""
        self.state = AgentState.PLANNING

        system_msg = ROLE_PROFILES.get(self.identity.role, {}).get("system_prompt", "You are a capable AI agent.")
        memory_context = self.memory.summarize_identity()

        messages = [{"role": "system", "content": f"{system_msg}\n\n[Memory]\n{memory_context}"}]
        if context:
            messages.append({"role": "system", "content": f"[Context from other agents]\n{context}"})
        messages.append({"role": "user", "content": task})

        # 注入可用工具
        tools_schema = self.tools.to_openai_format() if self.tools._tools else None

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 800,
                "temperature": 0.3,
            }
            if tools_schema:
                kwargs["tools"] = tools_schema
                kwargs["tool_choice"] = tool_choice

            resp = await self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            result = {
                "content": msg.content or "",
                "tool_calls": [],
            }

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            return result
        except Exception as e:
            logger.error("[%s] think failed: %s", self.identity.name, e)
            return {"content": f"Error: {e}", "tool_calls": []}

    async def act(self, task: str, context: str = "", max_tool_rounds: int = 3) -> TaskResult:
        """Agent 执行任务：ReAct 循环（思考→调用工具→观察→再思考→完成）"""
        start = time.time()
        self.state = AgentState.PLANNING
        self.task_count += 1
        tool_calls_made = 0

        task_id = f"{self.identity.name}-task-{self.task_count}"

        try:
            current_context = context

            for round_num in range(max_tool_rounds):
                result = await self.think(task, current_context)

                # 没有工具调用 → 最终回答
                if not result["tool_calls"]:
                    self.state = AgentState.DONE
                    latency = (time.time() - start) * 1000
                    self.total_latency_ms += latency

                    content = result["content"].strip()
                    self.memory.save_fact(
                        f"last_task_{self.identity.name}",
                        f"{task[:100]} -> {content[:200]}",
                        source=f"crew_agent.{self.identity.name}",
                    )

                    return TaskResult(
                        task_id=task_id,
                        agent_name=self.identity.name,
                        content=content,
                        success=True,
                        latency_ms=latency,
                        tool_calls_made=tool_calls_made,
                    )

                # 有工具调用 → 执行
                self.state = AgentState.TOOL_CALL
                tool_results = []

                for tc in result["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    try:
                        tool_output = self.tools.execute(tool_name, args)
                        tool_calls_made += 1
                    except Exception as e:
                        tool_output = f"Tool error: {e}"

                    tool_results.append({
                        "tool_name": tool_name,
                        "result": str(tool_output)[:500],
                    })

                # 将工具结果注入上下文
                tool_context = "Tool results:\n" + "\n".join(
                    f"  [{tr['tool_name']}] {tr['result'][:300]}"
                    for tr in tool_results
                )
                current_context = (current_context + "\n\n" + tool_context) if current_context else tool_context

            # 达到最大轮次 → 最后一次思考
            final_result = await self.think(task, current_context)
            self.state = AgentState.DONE
            latency = (time.time() - start) * 1000
            self.total_latency_ms += latency

            return TaskResult(
                task_id=task_id,
                agent_name=self.identity.name,
                content=final_result["content"].strip(),
                success=True,
                latency_ms=latency,
                tool_calls_made=tool_calls_made,
            )

        except Exception as e:
            self.state = AgentState.ERROR
            return TaskResult(
                task_id=task_id,
                agent_name=self.identity.name,
                content="",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
                tool_calls_made=tool_calls_made,
            )

    # ── 消息总线集成 ──

    async def listen(self, timeout: float = 1.0):
        """监听消息总线并处理"""
        if not self.bus:
            return

        envelope = await self.bus.receive(self.identity.name, timeout=timeout)
        if not envelope:
            return

        msg = envelope.message
        logger.info("[%s] received %s from %s", self.identity.name, msg.msg_type.value, msg.sender)

        if msg.msg_type == MsgType.DELEGATE:
            result = await self.act(str(msg.payload))
            self.bus.respond(msg.reply_to or envelope.id, result)

        elif msg.msg_type == MsgType.DIRECT:
            # 点对点：处理并可选回复
            result = await self.think(str(msg.payload))
            logger.info("[%s] processed direct message: %s", self.identity.name, result[:100])

        elif msg.msg_type == MsgType.BROADCAST:
            # 广播：接收信息，更新上下文
            logger.info("[%s] received broadcast: %s", self.identity.name, str(msg.payload)[:100])

    async def send_to(self, receiver: str, payload: Any) -> str:
        """向另一个 Agent 发送消息"""
        if not self.bus:
            raise RuntimeError("Not connected to message bus")
        return await self.bus.send(self.identity.name, receiver, payload)

    async def delegate_to(self, receiver: str, task: str) -> Any:
        """委派任务给另一个 Agent 并等待结果"""
        if not self.bus:
            raise RuntimeError("Not connected to message bus")
        return await self.bus.delegate(self.identity.name, receiver, task)

    # ── 状态查询 ──

    def stats(self) -> dict:
        return {
            "name": self.identity.name,
            "role": self.identity.role,
            "state": self.state.value,
            "tasks_completed": self.task_count,
            "avg_latency_ms": self.total_latency_ms / max(self.task_count, 1),
            "memory_facts": len(self.memory.facts),
        }

    def __repr__(self) -> str:
        return f"CrewAgent({self.identity.name}, {self.identity.role})"
