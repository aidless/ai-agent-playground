"""
LLM 客户端 —— 所有 Agent 共享的"电话线"。

支持三种调用模式：
  1. send()         —— 同步，等完整回复
  2. send_stream()  —— 同步流式，逐 token 输出
  3. asend()        —— 异步，等完整回复
  4. asend_stream() —— 异步流式，逐 token 输出

生产系统中流式输出是必选项——用户不能盯着白屏等 30 秒。
"""

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

_load_dotenv_done = False


def _ensure_dotenv():
    global _load_dotenv_done
    if not _load_dotenv_done:
        load_dotenv(Path(__file__).parent.parent / ".env")
        _load_dotenv_done = True


class LLMClient:
    """薄包装——认证、发送、提取文本。支持同步+异步+流式。"""

    def __init__(self):
        _ensure_dotenv()

        base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not base_url or not api_key:
            raise RuntimeError(
                "DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY must be set in .env file. "
                "Copy .env.example to .env and fill in your keys."
            )

        self._client = Anthropic(base_url=base_url, api_key=api_key)
        self._async_client = AsyncAnthropic(base_url=base_url, api_key=api_key)

    # ============================================================
    #  同步 API（现有代码不变）
    # ============================================================

    def send(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> str:
        """发消息给 AI，拿回完整文本回复。"""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return self._extract_text(response)

    # ============================================================
    #  同步流式 —— 逐 token 输出
    # ============================================================

    def send_stream(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> Generator[str, None, None]:
        """流式发送消息，逐 token yield 文本片段。

        使用方式：
            for chunk in llm.send_stream(messages, ...):
                print(chunk, end="", flush=True)
        """
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ============================================================
    #  异步 API —— 支持并发
    # ============================================================

    async def asend(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> str:
        """异步发送消息，拿回完整文本回复。"""
        response = await self._async_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return self._extract_text(response)

    async def asend_stream(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> AsyncGenerator[str, None]:
        """异步流式发送消息，逐 token yield 文本片段。

        使用方式：
            async for chunk in llm.asend_stream(messages, ...):
                print(chunk, end="", flush=True)
        """
        async with self._async_client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ============================================================
    #  内部工具
    # ============================================================

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in response.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "\n".join(parts) if parts else "[No text in response]"


# ============================================================
#  全局单例
# ============================================================

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
