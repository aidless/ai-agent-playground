"""
Agent 基类 —— 所有 AI Agent 的"骨架"。

Pipeline 模式（来自 HuggingFace Transformers）：
  preprocess → _forward → postprocess

支持三种调用模式：
  run()        —— 同步，返回完整结果
  run_stream() —— 同步流式，逐 token yield（生产必须）
  arun()       —— 异步，返回完整结果
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from .config import BaseAgentConfig
from .llm import get_client

# ============================================================
# 安全导入可选优化模块（缺失时自动降级，不阻断运行）
# ============================================================
try:
    from .security import get_input_validator, get_rate_limiter
    from .cache import get_llm_cache
    from .resilience import retry
    from .observability_enhanced import get_enhanced_tracer
    from .message_bus import publish
    _HAS_ENHANCEMENTS = True
except ImportError:
    _HAS_ENHANCEMENTS = False

    # ✅ 标准带参装饰器降级实现（必须三层结构）
    def retry(max_attempts=3, base_delay=1.0):
        def decorator(func):
            return func
        return decorator

    def publish(*args, **kwargs): pass

    class _DummyTracer:
        def start_span(self, name): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        status = "ok"
        error_message = ""
    def get_enhanced_tracer(): return _DummyTracer()

    def get_input_validator():
        return type('V', (), {'validate': lambda s, x: (True, ""), 'sanitize': lambda s, x: x})()
    def get_rate_limiter():
        return type('L', (), {'check': lambda s, x: (True, "")})()
    def get_llm_cache():
        return type('C', (), {'get': lambda s, *a: None, 'set': lambda s, *a: None})()

# ============================================================
#  Stream events
# ============================================================

@dataclass
class ToolCallEvent:
    """Agent 即将调用或已完成一次工具调用。"""
    phase: str  # "start" | "end"
    tool_name: str
    args: dict | None = None
    result: str | None = None

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
    """

    config_class: type[BaseAgentConfig] = BaseAgentConfig

    def __init__(self, config: BaseAgentConfig | None = None):
        self.config = config if config is not None else self.config_class()
        self.llm = get_client()

        # ✅ 统一初始化增强模块（移除无用的 self._retry）
        self._validator = get_input_validator()
        self._limiter = get_rate_limiter()
        self._cache = get_llm_cache()
        self._tracer = get_enhanced_tracer()
        self._publish = publish

    def run(self, inputs: Any, user_id: str = "anonymous", **kwargs) -> Any:
        """一键执行：安全检查 → 缓存 → 追踪 → 重试 → 执行"""
        allowed, reason = self._limiter.check(user_id)
        if not allowed:
            raise PermissionError(f"Rate limited: {reason}")

        if isinstance(inputs, str):
            valid, reason = self._validator.validate(inputs)
            if not valid:
                raise ValueError(f"Invalid input: {reason}")
            inputs = self._validator.sanitize(inputs)

        cache_key = str(inputs)
        cached = self._cache.get([{"role": "user", "content": cache_key}], self.config.model)
        if cached:
            self._publish("agent:cache_hit", {"agent": self.__class__.__name__})
            return cached

        with self._tracer.start_span(self.__class__.__name__) as span:
            try:
                result = self._run_with_retry(inputs, **kwargs)
                self._cache.set([{"role": "user", "content": cache_key}], self.config.model, result)
                self._publish("agent:complete", {"agent": self.__class__.__name__, "status": "success"})
                span.status = "ok"
                return result
            except Exception as e:
                self._publish("agent:error", {"agent": self.__class__.__name__, "error": str(e)})
                span.status = "error"
                span.error_message = str(e)
                raise

    # ✅ 使用模块级 retry，装饰器在类定义时正确解析
    @retry(max_attempts=3, base_delay=1.0)
    def _run_with_retry(self, inputs: Any, **kwargs) -> Any:
        """带重试的核心执行流程"""
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self._forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

    # ai_agent_playground/base.py 中的 BaseAgent 类内

    def run_stream(self, inputs: Any, user_id: str = "anonymous", **kwargs) -> Generator[StreamItem, None, None]:
        """流式执行：安全检查 → 追踪 → 流式生成（跳过缓存/重试）"""
        # 1. 速率限制 & 输入验证（与 run 一致）
        allowed, reason = self._limiter.check(user_id)
        if not allowed:
            raise PermissionError(f"Rate limited: {reason}")
        if isinstance(inputs, str):
            valid, reason = self._validator.validate(inputs)
            if not valid:
                raise ValueError(f"Invalid input: {reason}")
            inputs = self._validator.sanitize(inputs)

        # 2. 链路追踪（标记为流式）
        with self._tracer.start_span(f"{self.__class__.__name__}_stream") as span:
            try:
                model_inputs = self.preprocess(inputs, **kwargs)
                yield from self._forward_stream(model_inputs, **kwargs)
                span.status = "ok"
            except Exception as e:
                span.status = "error"
                span.error_message = str(e)
                raise
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

    def _forward_stream(self, model_inputs: dict[str, Any], **kwargs) -> Generator[StreamItem, None, None]:
        """流式 _forward。默认实现：完整回复作为一个 text delta 输出。"""
        output = self._forward(model_inputs, **kwargs)
        text = self.postprocess(output, **kwargs)
        if isinstance(text, str):
            yield text

    async def _forward_async(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """异步 _forward。默认实现：在线程池中跑同步版本。"""
        import asyncio
        return await asyncio.to_thread(self._forward, model_inputs, **kwargs)