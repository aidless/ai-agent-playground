"""Tool-Use Agent — ReAct loop with MCP-style tool calling.

Core pattern (ReAct: Reasoning + Acting):
  1. User asks a question
  2. LLM decides: answer directly OR call a tool
  3. If tool call → execute → feed result back to LLM
  4. LLM synthesizes final answer from tool results
  5. Repeat until answer or max rounds reached

This is the same pattern Claude Code, ChatGPT, and Copilot use internally.
"""

import json
import re
from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import MCPAgentConfig
from .tools import TOOLS, TOOL_DESCRIPTIONS


class MCPToolAgent(BaseAgent):
    """An agent that can use tools: search, read/write files, run commands, calculate.

    Pipeline:
        preprocess:   user message → format with tool list
        _forward:     ReAct loop: LLM decides → execute tool → feed back → repeat
        postprocess:  raw answer → clean formatted response
    """

    config_class = MCPAgentConfig

    def __init__(self, config: MCPAgentConfig | None = None):
        super().__init__(config)
        self.tools = TOOLS

    # ---- Pipeline ----

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        return {"question": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """ReAct loop: think → act → observe → think → ... → answer."""
        question = model_inputs["question"]
        conversation = [
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Available tools:\n" +
                "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) +
                f"\n\nTo use a tool, respond with JSON:\n"
                f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}\n'
                f"After getting tool results, give your final answer."
            )}
        ]

        tool_use_log = []

        for round_num in range(self.config.max_tool_rounds):
            # Call LLM
            reply = self.llm.send(
                messages=conversation,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
            )

            # Try to extract a tool call from the reply
            tool_call = self._parse_tool_call(reply)

            if tool_call is None:
                # No tool call → this is the final answer
                return {
                    "answer": reply,
                    "tool_rounds": round_num,
                    "tool_log": tool_use_log,
                }

            # Execute the tool
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            if tool_name not in self.tools:
                result = f"Unknown tool: {tool_name}. Available: {list(self.tools)}"
            else:
                try:
                    result = self.tools[tool_name](**tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"

            tool_use_log.append({
                "round": round_num + 1,
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:500],
            })

            # Feed tool result back to conversation
            conversation.append({"role": "assistant", "content": reply})
            conversation.append({"role": "user", "content": f"Tool result from {tool_name}:\n{result}\n\nBased on this, provide your answer."})

        # Max rounds reached — ask for final answer
        final = self.llm.send(
            messages=conversation + [{"role": "user", "content": "Please give your final answer now."}],
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
        """Format the result with tool use summary."""
        answer = model_outputs["answer"]
        log = model_outputs["tool_log"]

        if not log:
            return answer

        # Add a tool-use summary
        lines = [answer, "", "---", "*Tool calls made:*"]
        for entry in log:
            args_str = ", ".join(f"{k}={v}" for k, v in entry["args"].items())
            lines.append(f"- `{entry['tool']}({args_str})`")
            preview = entry["result"][:150].replace("\n", " ")
            lines.append(f"  → {preview}...")
        return "\n".join(lines)

    # ---- Tool call parser ----

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        """Extract a JSON tool call from LLM output.

        Handles both:
          {"tool": "web_search", "args": {"query": "Python LoRA"}}
          ```json\n{"tool": "web_search", ...}\n```
        """
        # Try JSON block first
        block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if block_match:
            try:
                return json.loads(block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try inline JSON object
        for match in re.finditer(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\}', text):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        return None

    # ---- High-level API ----

    def ask(self, question: str) -> str:
        """Ask a question — agent may use tools to answer."""
        return self.run(question)
