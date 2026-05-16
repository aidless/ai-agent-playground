"""HelloAgent config — declare what, not how."""

from dataclasses import dataclass
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class HelloAgentConfig(BaseAgentConfig):
    """Configuration for HelloAgent.

    Like BertConfig: declare typed params with defaults.
    The base class handles to_dict/from_dict/save/load.
    """

    agent_type: ClassVar[str] = "hello"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a helpful AI assistant. When you answer:\n"
        "- Be concise but thorough\n"
        "- Use examples when they help explain a concept\n"
        "- If you don't know something, say so honestly"
    )
