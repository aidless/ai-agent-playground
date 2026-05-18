"""AsyncAgent with reflection, learning, governance, and sandbox integration."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from agent.state import AgentContext, AgentState
from agent.tools.registry import ToolRegistry
from agent.async_llm_client import call_llm_stream_async
from agent.memory import get_memory, AgentMemory
from agent.governance import GovernancePanel
from agent.sandbox import SandboxExecutor
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
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.memory = memory or get_memory()
        self.governance = governance or GovernancePanel()
        self.sandbox = sandbox
        self.enable_reflection = enable_reflection
        self.enable_learning = enable_learning

    async def run(self, ctx: AgentContext, user_input: str) -> AgentContext:
        """Non-streaming execution: aggregate all events, return final context."""
        async for _ in self.run_stream(ctx, user_input):
            pass
        return ctx

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

        def _execute():
            if self.sandbox:
                sandbox_result = self.sandbox.execute(
                    tool_name, _raw_execute, args,
                    identity=getattr(ctx, "identity_id", "anonymous"),
                    session_id=ctx.trace_id,
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
        except Exception as e:
            result = f"Error: {e}"
            ctx.tool_results.append({"tool": tool_name, "result": result, "status": "error"})

        ctx.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        })

    async def _reflect_step(self, ctx: AgentContext) -> AsyncGenerator[dict, None]:
        """Self-reflection after tool calls."""
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
        except Exception as e:
            logger.warning("Reflection step failed (non-critical): %s", e)

    async def _learn_step(self, ctx: AgentContext, final_response: str) -> AsyncGenerator[dict, None]:
        """Extract lessons from execution and persist to memory."""
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
