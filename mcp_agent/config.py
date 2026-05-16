"""MCP Agent config."""

from dataclasses import dataclass
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class MCPAgentConfig(BaseAgentConfig):
    agent_type: ClassVar[str] = "mcp-agent"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    max_tool_rounds: int = 5
    mcp_command: list[str] | None = None

    system_prompt: str = (
        "You are an AI assistant with access to tools. "
        "Use tools when you need real-time information, file access, or computation.\n\n"
        "Tool use format:\n"
        "When you need to use a tool, respond with:\n"
        '{"tool": "tool_name", "args": {"arg1": "value1"}}\n\n'
        "After receiving tool results, continue with your answer.\n"
        "Only use tools when necessary. For general questions, answer directly."
    )
