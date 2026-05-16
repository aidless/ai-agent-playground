"""ResumeMatcherAgent — Pipeline-style resume analysis."""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import ResumeMatcherConfig
from .matcher import ResumeMatcher


class ResumeMatcherAgent(BaseAgent):
    """Analyze resume vs job description match.

    Pipeline:
        preprocess:   {resume, jd} → {resume_text, jd_text}
        _forward:     resume + jd → AI analysis → match report
        postprocess:  report → formatted output
    """

    config_class = ResumeMatcherConfig

    def __init__(self, config: ResumeMatcherConfig | None = None):
        super().__init__(config)
        self.matcher = ResumeMatcher(self.config, self.llm)

    def preprocess(self, inputs: dict, **kwargs) -> dict[str, Any]:
        """inputs = {'resume': str, 'jd': str}"""
        return {
            "resume_text": inputs["resume"],
            "jd_text": inputs["jd"],
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self.matcher.match(
            resume_text=model_inputs["resume_text"],
            jd_text=model_inputs["jd_text"],
        )
        return {"report": result.raw_report}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        return model_outputs["report"]

    def analyze(self, resume_text: str, jd_text: str) -> str:
        """One-line entry: give resume + JD, get match report."""
        return self.run({"resume": resume_text, "jd": jd_text})
