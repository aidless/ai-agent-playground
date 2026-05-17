"""
Agent 基类 —— 所有 AI Agent 的"骨架"。

Pipeline 模式（来自 HuggingFace Transformers）：
  preprocess → _forward → postprocess

<<<<<<< HEAD
支持三种调用模式：
  run()       —— 同步，返回完整结果
  run_stream() —— 同步流式，逐 token yield（生产必须）
  arun()      —— 异步，返回完整结果
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
=======
这个文件定义的就是"服务员工作流程"——三步走，永远不变。
具体的 Agent（HelloAgent、CodeReviewAgent...）只需要告诉这三步分别做什么。

这个设计模式来自 HuggingFace Transformers 源码的 Pipeline 类：
  preprocess（预处理）→ _forward（模型推理）→ postprocess（后处理）

集成了优化模块：
  - 安全控制（权限、输入验证、速率限制）
  - LLM缓存
  - 重试机制
  - 可观测性（链路追踪）
  - 消息总线
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
>>>>>>> 31ff4f2 (chore: commit pending local changes)
from typing import Any

from .config import BaseAgentConfig
from .llm import get_client
<<<<<<< HEAD


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
=======
from .security import get_input_validator, get_rate_limiter
from .cache import get_llm_cache
from .resilience import retry
from .observability_enhanced import get_enhanced_tracer
from .message_bus import publish
>>>>>>> 31ff4f2 (chore: commit pending local changes)


class BaseAgent(ABC):
    """所有 Agent 的骨架。

    子类实现 3 个方法：
      preprocess  → 把用户输入变成 AI 能理解的格式
      _forward    → 调用 AI
      postprocess → 把 AI 回复变成用户能理解的格式

<<<<<<< HEAD
    可选覆盖：
      _forward_stream → 流式调用 AI（返回 generator）
=======
    使用示例：
      class MyAgent(BaseAgent):
          def preprocess(self, inputs):
              return {"messages": [...]}

          def _forward(self, model_inputs):
              return {"reply": self.llm.send(...)})

          def postprocess(self, model_outputs):
              return model_outputs["reply"]
>>>>>>> 31ff4f2 (chore: commit pending local changes)
    """

    config_class: type[BaseAgentConfig] = BaseAgentConfig

    def __init__(self, config: BaseAgentConfig | None = None):
        self.config = config if config is not None else self.config_class()
        self.llm = get_client()

<<<<<<< HEAD
    # ============================================================
    #  同步完整调用（现有 API 不变）
    # ============================================================

    def run(self, inputs: Any, **kwargs) -> Any:
        """一键执行：preprocess → _forward → postprocess"""
=======
        # 集成优化模块
        self._validator = get_input_validator()
        self._limiter = get_rate_limiter()
        self._cache = get_llm_cache()
        self._tracer = get_enhanced_tracer()

    def run(self, inputs: Any, user_id: str = "anonymous", **kwargs) -> Any:
        """
        跑一遍完整流程：安全检查 → 缓存 → 追踪 → 重试 → 执行。
        """
        # 1. 速率限制
        allowed, reason = self._limiter.check(user_id)
        if not allowed:
            raise PermissionError(f"Rate limited: {reason}")

        # 2. 输入验证
        if isinstance(inputs, str):
            valid, reason = self._validator.validate(inputs)
            if not valid:
                raise ValueError(f"Invalid input: {reason}")
            inputs = self._validator.sanitize(inputs)

        # 3. 检查缓存
        cache_key = str(inputs)
        cached = self._cache.get([{"role": "user", "content": cache_key}], self.config.model)
        if cached:
            publish("agent:cache_hit", {"agent": self.__class__.__name__})
            return cached

        # 4. 链路追踪
        with self._tracer.start_span(self.__class__.__name__) as span:
            try:
                # 5. 执行（带重试）
                result = self._run_with_retry(inputs, **kwargs)

                # 存入缓存
                self._cache.set([{"role": "user", "content": cache_key}], self.config.model, result)

                # 发布完成消息
                publish("agent:complete", {"agent": self.__class__.__name__, "status": "success"})
                span.status = "ok"

                return result
            except Exception as e:
                publish("agent:error", {"agent": self.__class__.__name__, "error": str(e)})
                span.status = "error"
                span.error_message = str(e)
                raise

    @retry(max_attempts=3, base_delay=1.0)
    def _run_with_retry(self, inputs: Any, **kwargs) -> Any:
        """带重试的执行。"""
>>>>>>> 31ff4f2 (chore: commit pending local changes)
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self._forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

<<<<<<< HEAD
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
=======
    def run_stream(self, inputs: Any, **kwargs):
        """流式执行（不经过缓存和重试）。"""
        model_inputs = self.preprocess(inputs, **kwargs)
        yield from self._forward_stream(model_inputs, **kwargs)

    async def arun(self, inputs: Any, **kwargs) -> Any:
        """异步执行。"""
        import asyncio
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = await self._forward_async(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

    @abstractmethod
    def preprocess(self, inputs: Any, **kwargs) -> dict[str, Any]:
        """
        第1步：准备数据——把用户的原始输入变成 AI 能理解的格式。
        """
>>>>>>> 31ff4f2 (chore: commit pending local changes)
        ...

    @abstractmethod
    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
<<<<<<< HEAD
        """第2步：调用 AI（完整回复）"""
=======
        """
        第2步：调用 AI——把准备好的数据发过去，拿回原始结果。
        """
>>>>>>> 31ff4f2 (chore: commit pending local changes)
        ...

    @abstractmethod
    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> Any:
<<<<<<< HEAD
        """第3步：格式化输出"""
        ...

    # ============================================================
    #  可选覆盖 —— 流式 + 异步
    # ============================================================

    def _forward_stream(
        self, model_inputs: dict[str, Any], **kwargs
    ) -> Generator[StreamItem, None, None]:
        """流式 _forward。默认实现：完整回复作为一个 text delta 输出。"""
=======
        """
        第3步：格式化——把 AI 的原始回复变成用户喜欢的样子。
        """
        ...

    def _forward_stream(self, model_inputs: dict[str, Any], **kwargs):
        """流式 _forward。"""
>>>>>>> 31ff4f2 (chore: commit pending local changes)
        output = self._forward(model_inputs, **kwargs)
        text = self.postprocess(output, **kwargs)
        if isinstance(text, str):
            yield text

<<<<<<< HEAD
    async def _forward_async(
        self, model_inputs: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """异步 _forward。默认实现：在线程池中跑同步版本。"""
        import asyncio

        return await asyncio.to_thread(self._forward, model_inputs, **kwargs)
=======
    async def _forward_async(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """异步 _forward。"""
        import asyncio
        return await asyncio.to_thread(self._forward, model_inputs, **kwargs)
>>>>>>> 31ff4f2 (chore: commit pending local changes)
