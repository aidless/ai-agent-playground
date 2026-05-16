"""
CodeReviewAgent —— 用 AI 审查代码质量。

这是你写的第 2 个 Agent，演示了 Pipeline 模式如何串联多个组件。

工作流程（就像体检）：
  1. preprocess:  输入路径 → 扫描目录，找到所有代码文件（护士量身高体重）
  2. _forward:    每个文件发给 AI 审查（医生看化验单）
  3. postprocess: 审查结果 → Markdown 报告（总结健康报告）

三层分工：
  Scanner = 找文件的人（"去，把 src/ 下面所有 .py 文件找出来"）
  Reviewer = 审查文件的人（"这份代码有什么问题？"问 AI）
  Reporter = 写报告的人（把 AI 回复整理成漂亮的报告）
"""

from pathlib import Path
from typing import Any

from ai_agent_playground.base import BaseAgent  # Agent 骨架

from .config import CodeReviewConfig  # 审查配置（检查哪些语言、跳过哪些目录...）
from .report import ReportGenerator    # 报告生成器（AI 回复 → 漂亮 Markdown）
from .reviewer import Reviewer         # 审查器（文件 → AI 分析 → 问题列表）
from .scanner import Scanner           # 扫描器（目录 → 代码文件列表）


class CodeReviewAgent(BaseAgent):
    """
    AI 代码审查官。

    就像请了一个资深工程师审查你的代码：
      1. 告诉它"审查这个目录"
      2. 它找到所有代码文件
      3. 每个文件发给 AI 看一下
      4. 出报告：哪些有 bug、哪些不安全、哪些写得不好

    面试时可以说："我自己实现了一个 AI Code Review 工具，15+ 种语言，自动生成结构化报告。"
    """

    config_class = CodeReviewConfig

    def __init__(self, config: CodeReviewConfig | None = None):
        """创建审查官：准备好扫描器、审查器、报告器。"""
        super().__init__(config)
        # 三个"手下"，各司其职
        self.scanner = Scanner(self.config)       # 手下 1：找文件
        self.reviewer = Reviewer(self.config, self.llm)  # 手下 2：调 AI 审查
        self.reporter = ReportGenerator()         # 手下 3：写报告

    # ============================================================
    #  三步 Pipeline
    # ============================================================

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """
        第1步：扫描目录，找到所有要审查的代码文件。

        输入: "C:/my-project"（一个路径字符串）
        输出: {"root": "C:/my-project", "files": [FileInfo, FileInfo, ...]}

        就像：你告诉助手"去那个文件夹看看有什么文件"→ 助手回来给你一个清单。
        """
        root = inputs
        files = self.scanner.scan(root)  # 遍历目录，收集代码文件
        return {"root": root, "files": files}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        第2步：把每个文件发给 AI 审查。

        这是最耗时的一步——每个文件都要调一次 AI。
        如果文件很多，这个过程可能要好几分钟。

        AI 回复的结构（通过 prompt 约束的格式）：
          SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION
          比如：critical|45|bug|Off-by-one error|range(N) should be range(N-1)
        """
        files = model_inputs["files"]
        if not files:
            return {"results": [], "root": model_inputs["root"]}
        # 逐个发送给 AI，收集审查结果
        results = self.reviewer.review_files(files)
        return {"results": results, "root": model_inputs["root"]}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """
        第3步：把审查结果整理成一份漂亮的 Markdown 报告。

        报告结构：
          # Code Review Report: 项目名
          ## Summary（严重/警告/建议 各多少个）
          ## Critical Issues（必须先修的）
          ## Warnings（建议修的）
          ## Per-File Summary（每个文件的问题数）
        """
        results = model_outputs["results"]
        project_name = Path(model_outputs["root"]).name
        return self.reporter.generate(results, project_name)

    # ============================================================
    #  高级方法
    # ============================================================

    def review(self, path: str, output_dir: str | None = None) -> str:
        """
        一键审查：输入路径 → 出报告 → 保存到文件。

        返回保存的文件路径。

        就像：医生一键体检 → 出体检报告 → 打印出来给你。
        """
        report_md = self.run(path)  # 跑一遍完整流程

        # 默认报告保存位置：项目的 reports/ 目录
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "reports")

        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)  # 如果目录不存在就创建

        # 文件名带时间戳，避免覆盖
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        project_name = Path(path).name
        out_path = out_dir / f"review-{project_name}-{ts}.md"

        # 写到磁盘
        out_path.write_text(report_md, encoding="utf-8")
        return str(out_path)
