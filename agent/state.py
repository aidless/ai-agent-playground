"""Agent 状态机定义

状态流转：
  IDLE → PLANNING → TOOL_CALL → REFLECT → (PLANNING | DONE | LEARN)
                                                    ↓
                                                 (LEARN → DONE)

- IDLE: 初始状态，等待用户输入
- PLANNING: Agent 正在思考下一步行动
- TOOL_CALL: 正在并发执行工具
- REFLECT: 工具执行完毕后自我反思
- LEARN: 从本轮执行中提取教训
- DONE: 执行完成
- ERROR: 出现不可恢复的错误
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    REFLECT = "reflect"
    LEARN = "learn"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentContext:
    trace_id: str
    state: AgentState = AgentState.IDLE
    step: int = 0
    max_steps: int = 10  # 增加上限以支持反思步骤
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0
    latency: float = 0.0

    # 安全上下文
    identity: Any = None  # Identity object from request, checked by sandbox

    # MemGPT interrupt queue
    interrupts: list[dict[str, Any]] = field(default_factory=list)

    # 反思与学习
    reflections: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)

    # 轨迹记录（用于事后分析）
    trace_steps: list[dict[str, Any]] = field(default_factory=list)

    def record_step(self, event: str, data: dict[str, Any]):
        """记录一步执行轨迹"""
        self.trace_steps.append({
            "step": self.step,
            "state": self.state.value,
            "event": event,
            **data,
        })
