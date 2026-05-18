
from pydantic import BaseModel, ValidationError
from typing import Callable, Dict, Any
import logging, json

logger = logging.getLogger(__name__)

class ToolParamSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any]
    required: list[str] = []

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: ToolParamSchema
    func: Callable

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict, func: Callable):
        self._tools[name] = ToolDefinition(
            name=name, description=description,
            parameters=ToolParamSchema(**parameters), func=func
        )

    def to_openai_format(self) -> list:
        return [{"type": "function", "function": t.model_dump(exclude={"func"})} for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        try:
            return tool.func(**arguments)
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            raise
