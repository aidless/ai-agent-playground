"""Configuration system — like transformers.configuration_utils.PreTrainedConfig.

Every agent defines a config dataclass. Base class handles save/load/to_dict.
Declare parameters with types and defaults — the framework does the rest.
"""

from dataclasses import dataclass, fields
from typing import ClassVar


@dataclass
class BaseAgentConfig:
    """Base config for all agents. Subclass and declare agent-specific params.

    Like PreTrainedConfig: declare typed fields with defaults, and the
    framework handles serialization and validation automatically.

    Class attributes (override in subclasses):
        agent_type: str — unique identifier, e.g. "hello", "code-review"
    """

    agent_type: ClassVar[str] = "base"

    # LLM settings
    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful AI assistant."

    def to_dict(self) -> dict:
        """Serialize to dict (excludes ClassVar fields)."""
        result = {"agent_type": self.agent_type}
        for f in fields(self):
            if f.name not in ("agent_type",):
                result[f.name] = getattr(self, f.name)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "BaseAgentConfig":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**kwargs)
