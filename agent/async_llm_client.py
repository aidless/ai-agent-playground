import openai
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

logger = logging.getLogger(__name__)

async def call_llm_stream_async(
    client: openai.AsyncOpenAI,
    messages: list,
    model: str,
    trace_id: str,
    step: int,
    max_retries: int = 3
) -> AsyncGenerator[Dict[str, Any], None]:
    """异步流式调用 LLM，自带指数退避重试，隔离网络抖动"""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=30.0
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                yield {
                    "type": "chunk",
                    "content": delta.content or "",
                    "tool_calls": delta.tool_calls,
                    "finish_reason": chunk.choices[0].finish_reason
                }
            return  # 成功流式结束，退出重试循环
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            logger.warning(f"[{trace_id}] Step {step} LLM retry {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as e:
            logger.error(f"[{trace_id}] Step {step} LLM fatal error: {e}")
            raise