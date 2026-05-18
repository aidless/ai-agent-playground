"""AsyncAgent with reflection, learning, governance, sandbox integration,
and SuperAgent capabilities: reflect→action, debate, bootstrapping."""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

from agent.state import AgentContext, AgentState
from agent.tools.registry import ToolRegistry
from agent.async_llm_client import call_llm_stream_async
from agent.memory import get_memory, AgentMemory
from agent.governance import GovernancePanel
from agent.sandbox import SandboxExecutor
from agent.reflect_action import ReflectActionEngine
from agent.debate import DebateEngine, DebateResult
from agent.bootstrap import BootstrapEngine
from agent.evolution import PerformanceTracker, EvolutionEngine
from agent.meta_agent import MetaAgent
from observability.tracer import log_trace

logger = logging.getLogger(__name__)

REFLECT_PROMPT = (
    "You just executed tool calls and got results. Reflect briefly:\n"
    "1. Did the tools return what you expected?\n"
    "2. Do you have enough information to answer the user now?\n"
    "3. What should you do next?\n"
    "Keep your reflection to 1-2 sentences. Be concise."
)

LEARN_PROMPT = (
    "Review this interaction. What is one concrete lesson you can learn?\n"
    "Focus on: tool usage, error recovery, or how to be more efficient.\n"
    "Keep it to one sentence. If nothing to learn, say Nothing new."
)


class AsyncAgent:
    """Async Agent with streaming, tool calls, reflection, learning, and governance."""

    def __init__(
        self,
        client,
        model: str,
        registry: ToolRegistry,
        memory: Optional[AgentMemory] = None,
        governance: Optional[GovernancePanel] = None,
        sandbox: Optional[SandboxExecutor] = None,
        enable_reflection: bool = True,
        enable_learning: bool = True,
        enable_super_agent: bool = False,
        challenger_client=None,
        challenger_model: str = "",
        arbitrator_client=None,
        arbitrator_model: str = "",
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.memory = memory or get_memory()
        self.governance = governance or GovernancePanel()
        self.sandbox = sandbox
        self.enable_reflection = enable_reflection
        self.enable_learning = enable_learning
        self.enable_super_agent = enable_super_agent

        # SuperAgent engines
        self.reflect_action = ReflectActionEngine() if enable_super_agent else None
        self.bootstrap = BootstrapEngine(client, model) if enable_super_agent else None
        self.perf_tracker = PerformanceTracker() if enable_super_agent else None
        self.evolution = EvolutionEngine(client, self.perf_tracker, registry, model) if enable_super_agent else None
        self.meta_agent = MetaAgent(self, client, registry) if enable_super_agent else None
        self.debate_engine = (
            DebateEngine(client, challenger_client, arbitrator_client)
            if enable_super_agent and challenger_client
            else None
        )
        self.challenger_model = challenger_model
        self.arbitrator_model = arbitrator_model or model

    async def run(self, ctx: AgentContext, user_input: str) -> AgentContext:
        """Non-streaming execution: aggregate all events, return final context.

        After execution, MetaAgent observes and autonomously decides whether
        to evolve, bootstrap, degrade, or rollback tools.
        """
        async for _ in self.run_stream(ctx, user_input):
            pass
        # Autonomous self-improvement
        if self.meta_agent:
            await self.meta_agent.observe(ctx)
        return ctx

    async def debate_run(self, task: str, context: str = "") -> DebateResult:
        """SuperAgent: multi-model debate for complex tasks.

        Requires enable_super_agent=True and a challenger_client at init.
        Falls back to a simpler self-critique if only one model is available.
        """
        if self.debate_engine:
            return await self.debate_engine.debate(
                task=task,
                primary_model=self.model,
                challenger_model=self.challenger_model,
                arbitrator_model=self.arbitrator_model,
                context=context,
            )

        # Fallback: single-model self-critique
        result = DebateResult(
            debate_id=f"self-{id(task)}",
            task=task,
            primary_model=self.model,
            challenger_model=self.model,
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Solve this task. Then critique your own solution. "
                        "Finally, provide the improved solution after addressing the critique."
                    )},
                    {"role": "user", "content": task},
                ],
                max_tokens=2048,
            )
            result.consensus = response.choices[0].message.content or ""
            result.completed = True
        except Exception as e:
            result.error = str(e)
        return result

    def degrade_tool(self, tool_name: str) -> dict:
        """Manually degrade a tool (for testing/admin)."""
        if self.reflect_action:
            self.reflect_action._degrade_tool(tool_name, "manual degradation")
            return {"degraded": tool_name, "alternatives": self.reflect_action.get_alternatives(tool_name)}
        return {"error": "SuperAgent not enabled"}

    def get_super_status(self) -> dict:
        """Return SuperAgent subsystem status."""
        return {
            "reflect_action": self.reflect_action.status() if self.reflect_action else None,
            "debate": self.debate_engine.status() if self.debate_engine else None,
            "bootstrap": self.bootstrap.list_bootstrapped() if self.bootstrap else None,
            "evolution": self.evolution.status() if self.evolution else None,
            "performance": self.perf_tracker.all_metrics() if self.perf_tracker else None,
            "meta_agent": self.meta_agent.status() if self.meta_agent else None,
        }

    async def run_stream(
        self, ctx: AgentContext, user_input: str
    ) -> AsyncGenerator[dict, None]:
        """Streaming execution with reflection and learning."""
        ctx.messages.append({"role": "user", "content": user_input})
        log_trace(ctx.trace_id, ctx.step, "start", {"input": user_input})
        ctx.record_step("start", {"input": user_input})
        yield {"type": "status", "content": "processing..."}

        while ctx.state not in (AgentState.DONE, AgentState.ERROR) and ctx.step < ctx.max_steps:
            ctx.state = AgentState.PLANNING
            ctx.step += 1
            ctx.record_step("planning", {})

            if ctx.step == 1:
                self._inject_memory_context(ctx)

            yield {"type": "status", "content": "thinking...", "step": ctx.step}

            try:
                self._stream_result = None
                async for event in self._stream_and_collect(ctx):
                    yield event
                full_content, tool_calls_buffer = self._stream_result

                msg_dict = {
                    "role": "assistant",
                    "content": full_content if not tool_calls_buffer else None,
                }
                if tool_calls_buffer:
                    msg_dict["tool_calls"] = tool_calls_buffer
                ctx.messages.append(msg_dict)

                if tool_calls_buffer:
                    async for event in self._handle_tool_calls(ctx, tool_calls_buffer):
                        yield event
                    if self.enable_reflection:
                        async for event in self._reflect_step(ctx):
                            yield event
                    continue

                if self.enable_learning:
                    async for event in self._learn_step(ctx, full_content):
                        yield event

                ctx.state = AgentState.DONE
                ctx.record_step("done", {"response": full_content})
                log_trace(ctx.trace_id, ctx.step, "done", {"response": full_content})
                yield {"type": "done", "content": full_content}
                break

            except Exception as e:
                ctx.state = AgentState.ERROR
                ctx.record_step("error", {"msg": str(e)})
                log_trace(ctx.trace_id, ctx.step, "error", {"msg": str(e)})
                yield {"type": "error", "content": str(e)}
                break

        self._persist_trace(ctx)

    async def _stream_and_collect(self, ctx: AgentContext) -> AsyncGenerator[dict, None]:
        """Stream LLM response, yield chunks, store result in self._stream_result."""
        full_content = ""
        tool_calls_buffer = []

        async for chunk in call_llm_stream_async(
            self.client, ctx.messages, self.model, ctx.trace_id, ctx.step
        ):
            content = chunk["content"] or ""
            full_content += content

            if content:
                yield {"type": "chunk", "content": content}

            if chunk["tool_calls"]:
                for tc_delta in chunk["tool_calls"]:
                    idx = tc_delta.index
                    if len(tool_calls_buffer) <= idx:
                        tool_calls_buffer.append({
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {
                                "name": tc_delta.function.name or "",
                                "arguments": tc_delta.function.arguments or "",
                            },
                        })
                    else:
                        tool_calls_buffer[idx]["function"]["name"] += tc_delta.function.name or ""
                        tool_calls_buffer[idx]["function"]["arguments"] += tc_delta.function.arguments or ""

        self._stream_result = (full_content, tool_calls_buffer)

    async def _handle_tool_calls(self, ctx: AgentContext, tool_calls_buffer: list) -> AsyncGenerator[dict, None]:
        """Execute tool calls concurrently with governance wrapping."""
        ctx.state = AgentState.TOOL_CALL

        # Apply tool degradation filter (SuperAgent)
        if self.reflect_action:
            tool_calls_buffer = self.reflect_action.filter_degraded_tools(tool_calls_buffer)

        tool_names = [tc["function"]["name"] for tc in tool_calls_buffer]
        ctx.record_step("tool_call", {"tools": tool_names})
        yield {"type": "tool_call", "content": tool_names}

        tasks = []
        for tc in tool_calls_buffer:
            tool_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            log_trace(ctx.trace_id, ctx.step, "tool_call", {"tool": tool_name, "args": args})
            tasks.append(self._exec_tool_async(tc["id"], tool_name, args, ctx))

        await asyncio.gather(*tasks)
        yield {"type": "status", "content": "tool_calls completed, continuing..."}

    async def _exec_tool_async(self, tool_call_id: str, tool_name: str, args: dict, ctx: AgentContext):
        """Execute tool via sandbox -> governance panel (permission -> circuit breaker -> audit)."""
        tool_start = time.time()

        def _execute():
            if self.sandbox:
                sandbox_result = self.sandbox.execute(
                    tool_name, _raw_execute, args,
                    identity=getattr(ctx, "identity_id", "anonymous"),
                    session_id=ctx.trace_id,
                    identity_obj=getattr(ctx, "identity", None),
                )
                if not sandbox_result.success:
                    raise RuntimeError(sandbox_result.error)
                return sandbox_result.output
            return self.registry.execute(tool_name, args)

        def _raw_execute():
            return self.registry.execute(tool_name, args)

        try:
            result = await self.governance.wrap_tool_call(
                tool_name, args, _execute, ctx.trace_id,
            )
            ctx.tool_results.append({"tool": tool_name, "result": result, "status": "ok"})
            if self.perf_tracker:
                elapsed_ms = (time.time() - tool_start) * 1000
                self.perf_tracker.record(tool_name, success=True, latency_ms=elapsed_ms)
        except Exception as e:
            result = f"Error: {e}"
            ctx.tool_results.append({"tool": tool_name, "result": result, "status": "error"})
            if self.perf_tracker:
                elapsed_ms = (time.time() - tool_start) * 1000
                self.perf_tracker.record(tool_name, success=False, latency_ms=elapsed_ms, error=str(e))

        ctx.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        })

    async def _reflect_step(self, ctx: AgentContext) -> AsyncGenerator[dict, None]:
        """Self-reflection after tool calls. Triggers Reflect→Action loop."""
        ctx.state = AgentState.REFLECT
        ctx.record_step("reflect_start", {})

        reflect_msgs = [
            {"role": "system", "content": REFLECT_PROMPT},
            *ctx.messages[-4:],
        ]

        try:
            reflection = ""
            async for chunk in call_llm_stream_async(
                self.client, reflect_msgs, self.model, ctx.trace_id, ctx.step
            ):
                if chunk["content"]:
                    reflection += chunk["content"]

            reflection = reflection.strip()
            if reflection:
                ctx.reflections.append(reflection)
                ctx.record_step("reflect_result", {"reflection": reflection})
                yield {"type": "reflection", "content": reflection}

                # Reflect→Action: evaluate and act on reflection
                if self.reflect_action:
                    actions = self.reflect_action.evaluate(reflection, ctx.tool_results)
                    for action in actions:
                        yield {"type": "action", "content": action}
                        ctx.record_step("reflect_action", action)
                        if action["type"] == "missing_tool" and self.bootstrap:
                            yield {"type": "bootstrapping", "content": action["suggested_name"]}
        except Exception as e:
            logger.warning("Reflection step failed (non-critical): %s", e)

    async def _learn_step(self, ctx: AgentContext, final_response: str) -> AsyncGenerator[dict, None]:
        """Extract lessons from execution and persist to memory. Triggers bootstrapping."""
        ctx.state = AgentState.LEARN
        ctx.record_step("learn_start", {})

        learn_msgs = [
            {"role": "system", "content": LEARN_PROMPT},
            {"role": "user", "content": f"Conversation:\n{ctx.messages}\n\nFinal response: {final_response}"},
        ]

        try:
            lesson = ""
            async for chunk in call_llm_stream_async(
                self.client, learn_msgs, self.model, ctx.trace_id, ctx.step
            ):
                if chunk["content"]:
                    lesson += chunk["content"]

            lesson = lesson.strip()
            if lesson and lesson != "Nothing new.":
                ctx.lessons.append(lesson)
                self.memory.add_lesson(
                    lesson=lesson,
                    context=f"trace_id={ctx.trace_id}, step={ctx.step}",
                    success=(ctx.state != AgentState.ERROR),
                )
                yield {"type": "lesson", "content": lesson}

            # Skills bootstrapping: check for missing-tool actions from reflection
            if self.bootstrap:
                missing_actions = [
                    a for a in ctx.trace_steps
                    if a.get("event") == "reflect_action" and a.get("type") == "missing_tool"
                ]
                if missing_actions:
                    for action in missing_actions[-1:]:  # Only bootstrap the latest
                        suggested_name = action.get("suggested_name", "unknown_tool")
                        description = action.get("description", "")
                        yield {"type": "bootstrapping", "content": f"Generating tool: {suggested_name}"}
                        tool = await self.bootstrap.generate_from_reflection(description, suggested_name)
                        if tool.validated:
                            success = self.bootstrap.register_tool(tool, self.registry)
                            yield {"type": "tool_registered", "content": {"name": suggested_name, "success": success}}
        except Exception as e:
            logger.warning("Learn step failed (non-critical): %s", e)

    def _inject_memory_context(self, ctx: AgentContext):
        """Inject relevant memories as system context."""
        identity = self.memory.summarize_identity()
        if identity and identity != "## 我的记忆":
            ctx.messages.insert(0, {
                "role": "system",
                "content": f"[Memory Context]\n{identity}\n\nUse these past learnings to make better decisions.",
            })

    def _persist_trace(self, ctx: AgentContext):
        """Save execution trace to disk."""
        try:
            self.memory.save_trace(ctx.trace_id, ctx.trace_steps)
        except Exception as e:
            logger.warning("Failed to persist trace: %s", e)
