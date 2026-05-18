from agent.state import AgentContext, AgentState
from agent.tools.registry import ToolRegistry
from agent.llm_client import call_llm_with_retry
from observability.tracer import log_trace
import openai
import json
import logging

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, client: openai.Client, model: str, registry: ToolRegistry):
        self.client = client
        self.model = model
        self.registry = registry

    def run(self, ctx: AgentContext, user_input: str) -> AgentContext:
        ctx.messages.append({"role": "user", "content": user_input})
        log_trace(ctx.trace_id, ctx.step, "start", {"input": user_input})

        while ctx.state != AgentState.DONE and ctx.step < ctx.max_steps:
            ctx.state = AgentState.PLANNING
            ctx.step += 1
            log_trace(ctx.trace_id, ctx.step, "planning", {})

            try:
                resp = call_llm_with_retry(self.client, ctx.messages, self.model, ctx.trace_id, ctx.step)
                choice = resp.choices[0].message

                # 安全构造 assistant 消息（兼容各版本 openai SDK）
                msg_dict = {"role": "assistant", "content": choice.content}
                if choice.tool_calls:
                    msg_dict["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in choice.tool_calls
                    ]
                ctx.messages.append(msg_dict)

                if choice.tool_calls:
                    ctx.state = AgentState.TOOL_CALL
                    for tc in choice.tool_calls:
                        tool_name = tc.function.name
                        args = json.loads(tc.function.arguments)
                        log_trace(ctx.trace_id, ctx.step, "tool_call", {"tool": tool_name, "args": args})

                        result = self.registry.execute(tool_name, args)
                        ctx.tool_results.append({"tool": tool_name, "result": result})
                        ctx.messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
                else:
                    ctx.state = AgentState.DONE
                    log_trace(ctx.trace_id, ctx.step, "done", {"response": choice.content})

            except Exception as e:
                ctx.state = AgentState.ERROR
                log_trace(ctx.trace_id, ctx.step, "error", {"msg": str(e)})
                break

        return ctx