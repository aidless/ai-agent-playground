"""MCP Agent config."""

from dataclasses import dataclass
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class MCPAgentConfig(BaseAgentConfig):
    agent_type: ClassVar[str] = "mcp-agent"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    max_tool_rounds: int = 5  # Max rounds of tool calling before final answer

    system_prompt: str = (
        "You are an AI assistant with access to tools. "
        "Use tools when you need real-time information, file access, or computation.\n\n"
        "Tool use format:\n"
        "When you need to use a tool, respond with:\n"
        '{"tool": "tool_name", "args": {"arg1": "value1"}}\n\n'
        "After receiving tool results, continue with your answer.\n"
        "Only use tools when necessary. For general questions, answer directly.\n\n"
        "Available tools:\n"
        "- web_search: Search the internet for current information\n"
        "- read_file: Read a file from disk\n"
        "- write_file: Write content to a file\n"
        "- run_command: Execute a shell command and return output\n"
        "- calculator: Evaluate a mathematical expression"
    )
