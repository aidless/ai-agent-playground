"""CodeReviewAgent — Pipeline-style code review.

Pipeline: root path → scan → review (AI) → report (Markdown)
"""

from pathlib import Path
from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import CodeReviewConfig
from .report import ReportGenerator
from .reviewer import Reviewer
from .scanner import Scanner


class CodeReviewAgent(BaseAgent):
    """AI-powered code review agent.

    Pipeline:
        preprocess:   path → scan → list[FileInfo]
        _forward:     list[FileInfo] → AI review → list[ReviewResult]
        postprocess:  list[ReviewResult] → Markdown report
    """

    config_class = CodeReviewConfig

    def __init__(self, config: CodeReviewConfig | None = None):
        super().__init__(config)
        self.scanner = Scanner(self.config)
        self.reviewer = Reviewer(self.config, self.llm)
        self.reporter = ReportGenerator()

    # ---- Pipeline implementation ----

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """Scan directory → list of FileInfo."""
        root = inputs
        files = self.scanner.scan(root)
        return {"root": root, "files": files}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """AI review each file → list of ReviewResult."""
        files = model_inputs["files"]
        if not files:
            return {"results": [], "root": model_inputs["root"]}
        results = self.reviewer.review_files(files)
        return {"results": results, "root": model_inputs["root"]}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """ReviewResult list → Markdown report."""
        results = model_outputs["results"]
        project_name = Path(model_outputs["root"]).name
        return self.reporter.generate(results, project_name)

    # ---- Higher-level API ----

    def review(self, path: str, output_dir: str | None = None) -> str:
        """Run full review and save report. Returns report path."""
        report_md = self.run(path)

        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "reports")

        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        project_name = Path(path).name
        out_path = out_dir / f"review-{project_name}-{ts}.md"
        out_path.write_text(report_md, encoding="utf-8")

        return str(out_path)
