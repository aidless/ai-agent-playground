"""
Agent 基类 —— 所有 AI Agent 的"骨架"。

Pipeline 模式（来自 HuggingFace Transformers）：
  preprocess → _forward → postprocess

支持三种调用模式：
  run()       —— 同步，返回完整结果
  run_stream() —— 同步流式，逐 token yield（生产必须）
  arun()      —— 异步，返回完整结果
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from .config import BaseAgentConfig
from .llm import get_client


# ============================================================
#  Stream events
# ============================================================


@dataclass
class ToolCallEvent:
    """Agent 即将调用或已完成一次工具调用。"""

    phase: str  # "start" | "end"
    tool_name: str
    args: dict | None = None  # present on "start"
    result: str | None = None  # present on "end"


# A stream item is either a text delta or a tool event
StreamItem = str | ToolCallEvent


# ============================================================
#  BaseAgent
# ============================================================


class BaseAgent(ABC):
    """所有 Agent 的骨架。

    子类实现 3 个方法：
      preprocess  → 把用户输入变成 AI 能理解的格式
      _forward    → 调用 AI
      postprocess → 把 AI 回复变成用户能理解的格式

    可选覆盖：
      _forward_stream → 流式调用 AI（返回 generator）
    """

    config_class: type[BaseAgentConfig] = BaseAgentConfig

    def __init__(self, config: BaseAgentConfig | None = None):
        self.config = config if config is not None else self.config_class()
        self.llm = get_client()

    # ============================================================
    #  同步完整调用（现有 API 不变）
    # ============================================================

    def run(self, inputs: Any, **kwargs) -> Any:
        """一键执行：preprocess → _forward → postprocess"""
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self._forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

    # ============================================================
    #  同步流式调用
    # ============================================================

    def run_stream(self, inputs: Any, **kwargs) -> Generator[StreamItem, None, None]:
        """流式执行：逐 token 输出 + 工具调用事件。

        Yields:
            str           — 文本增量（token chunk）
            ToolCallEvent — 工具调用开始/结束事件

        使用方式：
            for item in agent.run_stream("What is 15*15?"):
                if isinstance(item, str):
                    print(item, end="", flush=True)
                elif isinstance(item, ToolCallEvent):
                    print(f"\n[{item.phase}] {item.tool_name}")
        """
        model_inputs = self.preprocess(inputs, **kwargs)
        yield from self._forward_stream(model_inputs, **kwargs)

    # ============================================================
    #  异步调用
    # ============================================================

    async def arun(self, inputs: Any, **kwargs) -> Any:
        """异步一键执行。"""
        import asyncio

        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = await self._forward_async(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

    # ============================================================
    #  抽象方法 —— 子类必须实现
    # ============================================================

    @abstractmethod
    def preprocess(self, inputs: Any, **kwargs) -> dict[str, Any]:
        """第1步：准备数据"""
        ...

    @abstractmethod
    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """第2步：调用 AI（完整回复）"""
        ...

    @abstractmethod
    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> Any:
        """第3步：格式化输出"""
        ...

    # ============================================================
    #  可选覆盖 —— 流式 + 异步
    # ============================================================

    def _forward_stream(
        self, model_inputs: dict[str, Any], **kwargs
    ) -> Generator[StreamItem, None, None]:
        """流式 _forward。默认实现：完整回复作为一个 text delta 输出。"""
        output = self._forward(model_inputs, **kwargs)
        text = self.postprocess(output, **kwargs)
        if isinstance(text, str):
            yield text

    async def _forward_async(
        self, model_inputs: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """异步 _forward。默认实现：在线程池中跑同步版本。"""
        import asyncio

        return await asyncio.to_thread(self._forward, model_inputs, **kwargs)
