import pytest
from agent.tools.registry import ToolRegistry
from agent.state import AgentContext

@pytest.fixture
def tool_registry():
    reg = ToolRegistry()
    reg.register("calc", "加法计算", {"properties": {"a": {"type": "int"}, "b": {"type": "int"}}, "required": ["a", "b"]}, lambda a, b: a + b)
    return reg

@pytest.fixture
def fresh_context():
    return AgentContext(trace_id="pytest_001", max_steps=3)