"""对话上下文压缩器 — 借鉴 Hermes Agent 的压缩策略

三种压缩模式:
    1. 简单截断 (truncate) — 保留首条 + 最近 N 条
    2. LLM 摘要 (summarize) — 用 LLM 将前半段压缩为摘要
    3. 混合模式 (hybrid) — 摘要 + 最近完整消息

用法:
    compressor = ContextCompressor(llm_client)
    compressed = await compressor.compress(messages, mode="hybrid")
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ContextCompressor:
    """对话上下文压缩器

    当上下文接近模型限制时自动压缩历史对话，
    保留关键信息同时释放 token 预算。
    """

    def __init__(self, llm_client, model: str = "deepseek-chat"):
        self.client = llm_client
        self.model = model

    async def compress(
        self,
        messages: list[dict],
        max_recent: int = 4,
        mode: str = "hybrid",
    ) -> list[dict]:
        """压缩对话历史

        Args:
            messages: 原始消息列表
            max_recent: 保留最近 N 条完整消息
            mode: truncate | summarize | hybrid

        Returns:
            压缩后的消息列表
        """
        if len(messages) <= max_recent + 2:
            return messages  # 不需要压缩

        if mode == "truncate":
            return self._truncate(messages, max_recent)
        elif mode == "summarize":
            return await self._summarize(messages, max_recent)
        elif mode == "hybrid":
            return await self._hybrid(messages, max_recent)
        return messages

    @staticmethod
    def _truncate(messages: list[dict], max_recent: int) -> list[dict]:
        """简单截断：保留首条系统消息 + 最近 N 条"""
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= max_recent:
            return messages

        truncated = system_messages + non_system[-max_recent:]
        logger.info("截断压缩: %d → %d 条消息", len(messages), len(truncated))
        return truncated

    async def _summarize(self, messages: list[dict], max_recent: int) -> list[dict]:
        """LLM 摘要：将前半段压缩为一条摘要消息"""
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= max_recent:
            return messages

        to_summarize = non_system[:-max_recent]
        recent = non_system[-max_recent:]

        summary = await self._llm_summarize(to_summarize)
        if not summary:
            return self._truncate(messages, max_recent)  # 回退

        compressed = system_messages + [
            {"role": "system", "content": f"[Conversation summary]\n{summary}"}
        ] + recent

        logger.info("摘要压缩: %d → %d 条消息", len(messages), len(compressed))
        return compressed

    async def _hybrid(self, messages: list[dict], max_recent: int) -> list[dict]:
        """混合模式：LLM 摘要 + 关键工具调用保留"""
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= max_recent:
            return messages

        to_summarize = non_system[:-max_recent]
        recent = non_system[-max_recent:]

        # 从要压缩的消息中提取关键工具调用
        key_tools = []
        for m in to_summarize:
            if m.get("role") == "tool" or m.get("tool_calls"):
                key_tools.append(m)

        summary = await self._llm_summarize(to_summarize)
        if not summary:
            return self._truncate(messages, max_recent)

        compressed = system_messages + [
            {"role": "system", "content": f"[Conversation summary — past interactions]\n{summary}"}
        ]

        # 保留关键工具交互
        if len(key_tools) <= 4:
            compressed.extend(key_tools)

        compressed.extend(recent)

        logger.info("混合压缩: %d → %d 条消息 (保留 %d 条关键工具)", len(messages), len(compressed), len(key_tools))
        return compressed

    async def _llm_summarize(self, messages: list[dict]) -> Optional[str]:
        """调用 LLM 生成摘要"""
        if not messages:
            return None

        # 只提取关键信息
        brief = []
        for m in messages:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:300]
            if content:
                brief.append(f"[{role}] {content}")

        prompt = (
            "Summarize this conversation fragment in 2-3 sentences. "
            "Focus on: what was asked, what was done, what tools were used, and the outcome.\n\n"
            "Fragment:\n" + "\n".join(brief[-20:])
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("LLM 摘要生成失败: %s", e)
            return None
