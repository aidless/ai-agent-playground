import pytest
from unittest.mock import MagicMock, patch
from agent.core import Agent
from agent.state import AgentState


def _make_llm_response(content=None, tool_calls=None, tokens=10):
    """构造 Mock LLM 响应"""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_msg.tool_calls = tool_calls
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_resp.usage.total_tokens = tokens
    return mock_resp


@patch("agent.core.call_llm_with_retry")
def test_direct_answer(mock_llm, tool_registry, fresh_context):
    mock_llm.return_value = _make_llm_response(content="直接回答")
    agent = Agent(client=MagicMock(), model="test", registry=tool_registry)
    ctx = agent.run(fresh_context, "你好")
    assert ctx.state == AgentState.DONE
    assert ctx.step == 1


@patch("agent.core.call_llm_with_retry")
def test_tool_call_flow(mock_llm, tool_registry, fresh_context):
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "calc"
    tc.function.arguments = '{"a": 2, "b": 3}'

    mock_llm.side_effect = [
        _make_llm_response(tool_calls=[tc]),
        _make_llm_response(content="计算完成")
    ]
    agent = Agent(client=MagicMock(), model="test", registry=tool_registry)
    ctx = agent.run(fresh_context, "算一下2+3")

    assert ctx.state == AgentState.DONE
    assert ctx.step == 2
    assert len(ctx.tool_results) == 1
    assert ctx.tool_results[0]["result"] == 5