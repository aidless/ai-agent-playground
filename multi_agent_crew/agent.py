"""CrewAgent — multi-agent collaboration, Pipeline-style.

Pipeline: requirement → PM tasks → Dev code → QA review → DevOps deploy
"""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import CrewConfig
from .crew import Crew


class CrewAgent(BaseAgent):
    """Orchestrates a full development crew from a single requirement.

    Like a high-level Pipeline: one sentence in, a complete project out.
    """

    config_class = CrewConfig

    def __init__(self, config: CrewConfig | None = None):
        super().__init__(config)
        self.crew = Crew(self.config, self.llm)

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        return {"requirement": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self.crew.run(model_inputs["requirement"])
        return {
            "requirement": result.requirement,
            "tasks": result.tasks,
            "code": result.code,
            "qa_report": result.qa_report,
            "devops_config": result.devops_config,
        }

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """Format the full crew output as a readable report."""
        lines = [
            "# Multi-Agent Crew Report",
            "",
            f"## Requirement",
            f"> {model_outputs['requirement']}",
            "",
            "## Task Breakdown (PM)",
            "",
        ]
        for t in model_outputs["tasks"]:
            lines.append(f"- **[{t['id']}]** ({t['priority']}) {t['title']}")
            lines.append(f"  {t['description']}")

        lines.extend(["", "## Generated Code (Dev)", ""])
        for tid, code in model_outputs["code"].items():
            lines.append(f"### Task {tid}")
            lines.append("")
            lines.append(code)
            lines.append("")

        if model_outputs["qa_report"]:
            lines.extend(["## QA Review", "", model_outputs["qa_report"], ""])

        if model_outputs["devops_config"]:
            lines.extend(["## Deployment Config (DevOps)", "", model_outputs["devops_config"], ""])

        return "\n".join(lines)

    def build(self, requirement: str) -> str:
        """One-line entry: give a requirement, get a complete project report."""
        return self.run(requirement)
