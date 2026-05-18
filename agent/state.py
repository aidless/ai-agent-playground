
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any

class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"

@dataclass
class AgentContext:
    trace_id: str
    state: AgentState = AgentState.IDLE
    step: int = 0
    max_steps: int = 5
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0
    latency: float = 0.0
