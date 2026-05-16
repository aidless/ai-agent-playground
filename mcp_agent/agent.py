"""
MCP Tool-Use Agent — 能通过 MCP 协议动态发现和使用工具的 AI Agent。

支持两种模式：
  1. MCP 模式：连接外部 MCP Server，动态发现工具、通过 MCP 协议调用
  2. 本地模式：不连 MCP Server 时，使用内置的 5 个工具

MCP 协议让 Agent 不再硬编码工具——任何实现 MCP 的服务都能直接接入。

核心机制：ReAct 循环（Reasoning + Acting = 推理 + 行动）
"""

import json
import re
from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import MCPAgentConfig
from .mcp_client import MCPClient
from .tools import TOOLS, TOOL_DESCRIPTIONS


class MCPToolAgent(BaseAgent):
    """能使用工具的 Agent，支持 MCP 协议动态工具发现。

    工作流程（ReAct 循环）：
      preprocess:  发现工具 → 构建工具描述 → 组装对话
      _forward:    ReAct 循环（最长 max_tool_rounds 轮）
      postprocess: 格式化答案 + 工具使用摘要
    """

    config_class = MCPAgentConfig

    def __init__(self, config: MCPAgentConfig | None = None):
        super().__init__(config)
        self._mcp: MCPClient | None = None
        self._remote_tools: dict[str, dict] = {}

        if self.config.mcp_command:
            self._mcp = MCPClient(self.config.mcp_command)
            self._mcp.start()
            self._discover_tools()

    # ============================================================
    #  工具发现
    # ============================================================

    def _discover_tools(self):
        """通过 MCP 协议动态发现工具列表。"""
        tools = self._mcp.list_tools()
        self._remote_tools = {}
        for t in tools:
            self._remote_tools[t["name"]] = {
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
            }

    # ============================================================
    #  三步 Pipeline
    # ============================================================

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        return {"question": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        question = model_inputs["question"]

        if self._mcp:
            tool_desc_text = "\n".join(
                f"- {name}: {info['description']}"
                for name, info in self._remote_tools.items()
            )
        else:
            tool_desc_text = "\n".join(
                f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
            )

        conversation = [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Available tools:\n"
                    f"{tool_desc_text}\n\n"
                    f'To use a tool, respond with JSON:\n'
                    f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}\n'
                    f"After getting tool results, give your final answer."
                ),
            }
        ]

        tool_use_log = []

        for round_num in range(self.config.max_tool_rounds):
            reply = self.llm.send(
                messages=conversation,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
            )

            tool_call = self._parse_tool_call(reply)
            if tool_call is None:
                return {
                    "answer": reply,
                    "tool_rounds": round_num,
                    "tool_log": tool_use_log,
                }

            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            result = self._execute_tool(tool_name, tool_args)

            tool_use_log.append({
                "round": round_num + 1,
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:500],
            })

            conversation.append({"role": "assistant", "content": reply})
            conversation.append({
                "role": "user",
                "content": f"Tool result from {tool_name}:\n{result}\n\nBased on this, provide your answer.",
            })

        final = self.llm.send(
            messages=conversation
            + [{"role": "user", "content": "Please give your final answer now."}],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
        )
        return {
            "answer": final,
            "tool_rounds": self.config.max_tool_rounds,
            "tool_log": tool_use_log,
        }

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        answer = model_outputs["answer"]
        log = model_outputs["tool_log"]

        if not log:
            return answer

        lines = [answer, "", "---", "*Tool calls made:*"]
        for entry in log:
            args_str = ", ".join(f"{k}={v}" for k, v in entry["args"].items())
            lines.append(f"- `{entry['tool']}({args_str})`")
            preview = entry["result"][:150].replace("\n", " ")
            lines.append(f"  → {preview}...")
        return "\n".join(lines)

    # ============================================================
    #  工具执行
    # ============================================================

    def _execute_tool(self, name: str, args: dict) -> str:
        if self._mcp and name in self._remote_tools:
            try:
                return self._mcp.call_tool(name, args)
            except Exception as e:
                return f"MCP tool error: {e}"

        if name in TOOLS:
            try:
                return TOOLS[name](**args)
            except Exception as e:
                return f"Tool error: {e}"

        available = (
            list(self._remote_tools) if self._mcp else list(TOOLS)
        )
        return f"Unknown tool: {name}. Available: {available}"

    # ============================================================
    #  JSON 解析器
    # ============================================================

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if block_match:
            try:
                return json.loads(block_match.group(1))
            except json.JSONDecodeError:
                pass

        for match in re.finditer(
            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\}',
            text,
        ):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        return None

    # ============================================================
    #  流式 ReAct 循环
    # ============================================================

    def _forward_stream(
        self, model_inputs: dict[str, Any], **kwargs
    ):
        """流式 ReAct 循环：逐 token 输出 + 工具调用事件。

        Yields:
            str — 文本片段
            ToolCallEvent — 工具调用开始/结束
        """
        from ai_agent_playground.base import ToolCallEvent

        question = model_inputs["question"]

        if self._mcp:
            tool_desc_text = "\n".join(
                f"- {name}: {info['description']}"
                for name, info in self._remote_tools.items()
            )
        else:
            tool_desc_text = "\n".join(
                f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
            )

        conversation = [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Available tools:\n"
                    f"{tool_desc_text}\n\n"
                    f'To use a tool, respond with JSON:\n'
                    f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}\n'
                    f"After getting tool results, give your final answer."
                ),
            }
        ]

        for round_num in range(self.config.max_tool_rounds):
            # Stream the LLM response, collecting full text
            full_reply = ""
            for chunk in self.llm.send_stream(
                messages=conversation,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
            ):
                full_reply += chunk
                yield chunk

            # Check for tool call in the collected response
            tool_call = self._parse_tool_call(full_reply)
            if tool_call is None:
                return  # Final answer already streamed

            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            yield ToolCallEvent(
                phase="start", tool_name=tool_name, args=tool_args
            )

            result = self._execute_tool(tool_name, tool_args)

            yield ToolCallEvent(
                phase="end", tool_name=tool_name, result=str(result)[:500]
            )

            conversation.append({"role": "assistant", "content": full_reply})
            conversation.append({
                "role": "user",
                "content": f"Tool result from {tool_name}:\n{result}\n\nBased on this, provide your answer.",
            })

        # Max rounds reached — stream final answer
        final_chunks = self.llm.send_stream(
            messages=conversation
            + [{"role": "user", "content": "Please give your final answer now."}],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
        )
        yield from final_chunks

    def ask(self, question: str) -> str:
        return self.run(question)

    def close(self):
        if self._mcp:
            self._mcp.close()
            self._mcp = None
