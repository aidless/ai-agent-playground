"""AI matching engine: resume + JD → match report."""

from dataclasses import dataclass

from ai_agent_playground.llm import LLMClient, get_client

from .config import ResumeMatcherConfig


@dataclass
class MatchResult:
    raw_report: str
    resume_length: int
    jd_length: int


class ResumeMatcher:
    """Compares a resume against a job description using AI."""

    def __init__(self, config: ResumeMatcherConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm or get_client()

    def match(self, resume_text: str, jd_text: str) -> MatchResult:
        """Run the full matching analysis."""
        # Truncate if too long
        resume_snippet = resume_text[:4000] if len(resume_text) > 4000 else resume_text
        jd_snippet = jd_text[:3000] if len(jd_text) > 3000 else jd_text

        user_msg = (
            f"## Job Description\n{jd_snippet}\n\n"
            f"## Candidate Resume\n{resume_snippet}\n\n"
            f"Please analyze the match between this resume and the job description."
        )

        report = self.llm.send(
            messages=[{"role": "user", "content": user_msg}],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
        )

        return MatchResult(
            raw_report=report,
            resume_length=len(resume_text),
            jd_length=len(jd_text),
        )
