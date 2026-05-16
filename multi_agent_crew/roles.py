"""Four role agents: PM, Dev, QA, DevOps — each a BaseAgent with a role prompt.

Like having different model heads in transformers (BertForQA, BertForClassification).
"""

from typing import Any

from ai_agent_playground.base import BaseAgent
from ai_agent_playground.config import BaseAgentConfig


class ProductManagerAgent(BaseAgent):
    """Breaks user requirement into concrete technical tasks."""

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": (
                f"User requirement: {inputs}\n\n"
                f"Break this down into at most {self.config.max_tasks} tasks."
            )}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.pm_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {"reply": self.llm.send(**model_inputs)}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> list[dict]:
        """Parse task list from AI output."""
        text = model_outputs["reply"]
        tasks = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            tasks.append({
                "id": parts[0].strip(),
                "priority": parts[1].strip(),
                "title": parts[2].strip(),
                "description": parts[3].strip(),
            })
        return tasks


class DeveloperAgent(BaseAgent):
    """Writes code for a given technical task."""

    def preprocess(self, inputs: dict, **kwargs) -> dict[str, Any]:
        """inputs = {'title': str, 'description': str}"""
        task = f"Task: {inputs['title']}\n\n{inputs['description']}"
        return {
            "messages": [{"role": "user", "content": task}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.dev_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {"reply": self.llm.send(**model_inputs)}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> dict:
        """Return the generated code."""
        return {"code": model_outputs["reply"]}


class QAAgent(BaseAgent):
    """Reviews code for bugs, security, and style issues."""

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """inputs = full code string to review"""
        return {
            "messages": [{"role": "user", "content": (
                f"Review the following code:\n\n{inputs}"
            )}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.qa_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {"reply": self.llm.send(**model_inputs)}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        return model_outputs["reply"]


class DevOpsAgent(BaseAgent):
    """Generates Dockerfile and deployment config."""

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """inputs = full project code string"""
        return {
            "messages": [{"role": "user", "content": (
                f"Generate deployment config for this project:\n\n{inputs}"
            )}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.devops_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {"reply": self.llm.send(**model_inputs)}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        return model_outputs["reply"]
