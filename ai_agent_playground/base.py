"""Base agent class — like transformers.pipelines.base.Pipeline.

Every agent follows the same template method:
    run(input) → preprocess → _forward → postprocess → result

Subclasses only need to implement:
    - preprocess(raw_input) → dict[str, Any]    (input → model-ready data)
    - _forward(prepared) → dict[str, Any]       (call the model)
    - postprocess(raw_output) → Any             (model output → user-facing result)
"""

from abc import ABC, abstractmethod
from typing import Any

from .config import BaseAgentConfig
from .llm import get_client


class BaseAgent(ABC):
    """Base class for all agents. Implements the Pipeline template method.

    Usage:
        class MyAgent(BaseAgent):
            def preprocess(self, inputs): ...
            def _forward(self, model_inputs): ...
            def postprocess(self, model_outputs): ...
    """

    # Override in subclasses to declare default config class
    config_class: type[BaseAgentConfig] = BaseAgentConfig

    def __init__(self, config: BaseAgentConfig | None = None):
        self.config = config if config is not None else self.config_class()
        self.llm = get_client()

    def run(self, inputs: Any, **kwargs) -> Any:
        """Run the full pipeline: preprocess → _forward → postprocess.

        Like Pipeline.run_single() — the main entry point.
        """
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self._forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)

    @abstractmethod
    def preprocess(self, inputs: Any, **kwargs) -> dict[str, Any]:
        """Transform raw user input into model-ready format.

        Like Pipeline.preprocess() — tokenize, validate, format.
        Must return a dict (model_inputs).
        """
        ...

    @abstractmethod
    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run the model/API call. Core inference step.

        Like Pipeline._forward() — the hot path. Keep this thin.
        Must return a dict (model_outputs).
        """
        ...

    @abstractmethod
    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> Any:
        """Transform raw model output into user-facing result.

        Like Pipeline.postprocess() — decode, format, clean up.
        """
        ...
