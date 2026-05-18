import pytest
import asyncio
from unittest.mock import MagicMock, patch
from agent.async_core import AsyncAgent
from agent.state import AgentState


def _make_async_stream(chunks_data):
    """模拟异步流式生成器"""

    async def stream_gen():
        for data in chunks_data:
            yield data

    return stream_gen()


def _make_func_mock(name_val, args_val):
    """✅ 正确构造 function mock，避开 MagicMock(name=...) 的保留字陷阱"""
    m = MagicMock()
    m.name = name_val
    m.arguments = args_val
    return m


@pytest.mark.asyncio
@patch("agent.async_core.call_llm_stream_async")
async def test_async_direct_answer(mock_stream, tool_registry, fresh_context):
    answer = _make_async_stream([
        {"type": "chunk", "content": "Hello", "tool_calls": None, "finish_reason": None},
        {"type": "chunk", "content": " World", "tool_calls": None, "finish_reason": "stop"}
    ])
    learn = _make_async_stream([
        {"type": "chunk", "content": "Nothing new.", "tool_calls": None, "finish_reason": "stop"}
    ])
    mock_stream.side_effect = [answer, learn]

    agent = AsyncAgent(client=MagicMock(), model="test", registry=tool_registry)
    ctx = await agent.run(fresh_context, "Hi")
    assert ctx.state == AgentState.DONE
    assert ctx.step == 1
    assert "Hello World" in ctx.messages[-1]["content"]


@pytest.mark.asyncio
@patch("agent.async_core.call_llm_stream_async")
async def test_async_concurrent_tools(mock_stream, tool_registry, fresh_context):
    # 🔧 构造工具调用分片
    tc1_d1 = MagicMock(index=0, id="call_1", function=_make_func_mock("calc", '{"a":1'))
    tc1_d2 = MagicMock(index=0, id=None, function=_make_func_mock(None, ', "b":2}'))
    tc2_d1 = MagicMock(index=1, id="call_2", function=_make_func_mock("calc", '{"a":5'))
    tc2_d2 = MagicMock(index=1, id=None, function=_make_func_mock(None, ', "b":5}'))

    # 🔄 第一轮：LLM 发起并行工具调用
    turn1 = _make_async_stream([
        {"type": "chunk", "content": None, "tool_calls": [tc1_d1, tc2_d1], "finish_reason": None},
        {"type": "chunk", "content": None, "tool_calls": [tc1_d2, tc2_d2], "finish_reason": "tool_calls"}
    ])
    # 🔄 第二轮：LLM 根据工具结果生成最终回答
    turn2 = _make_async_stream([
        {"type": "chunk", "content": "计算结果分别为 3 和 10。", "tool_calls": None, "finish_reason": "stop"}
    ])

    # 加第三个 mock 供 reflect 步骤使用（新版 AsyncAgent 会先反思再继续）
    turn_reflect = _make_async_stream([
        {"type": "chunk", "content": "Tools worked, proceed.", "tool_calls": None, "finish_reason": "stop"}
    ])
    mock_stream.side_effect = [turn1, turn_reflect, turn2]

    agent = AsyncAgent(client=MagicMock(), model="test", registry=tool_registry)
    ctx = await agent.run(fresh_context, "算 1+2 和 5+5")

    # 验证完整闭环：状态应为 DONE，且工具结果已正确捕获
    assert ctx.state == AgentState.DONE
    assert ctx.step == 2
    assert len(ctx.tool_results) == 2
    results = [r["result"] for r in ctx.tool_results]
    assert 3 in results and 10 in results
    assert "计算结果分别为 3 和 10。" in ctx.messages[-1]["content"]