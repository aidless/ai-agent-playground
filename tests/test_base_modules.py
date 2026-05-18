
import pytest
from agent.tools.registry import ToolRegistry
from agent.state import AgentContext, AgentState

def test_tool_registry():
    reg = ToolRegistry()
    reg.register("dummy", "测试工具", {"properties": {"x": {"type": "integer"}}, "required": ["x"]}, lambda x: x*2)
    assert reg.execute("dummy", {"x": 5}) == 10
    assert len(reg.to_openai_format()) == 1

def test_agent_context():
    ctx = AgentContext(trace_id="test001", max_steps=3)
    assert ctx.state == AgentState.IDLE
    assert ctx.step == 0
