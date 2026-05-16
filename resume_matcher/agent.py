"""
ResumeMatcherAgent —— AI 简历匹配分析器。

这是你写的第 5 个项目，也是 2026 年最热门的 AI 入门项目之一。

它做什么？
  上传简历 + 粘贴职位描述 → AI 分析匹配度 → 告诉你：
    - 匹配度多少%
    - 哪些关键词已经匹配
    - 哪些关键词缺失（JD 要求了但你简历里没写！）
    - 简历有什么漏洞
    - 具体的改进建议

就像：找一个职业顾问，帮你对比简历和职位要求，告诉你怎么改。

Pipeline:
  preprocess:  接收简历文本 + 职位描述
  _forward:    发给 AI 做匹配分析
  postprocess: 格式化报告
"""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import ResumeMatcherConfig
from .matcher import ResumeMatcher  # 匹配引擎（简历 vs JD）


class ResumeMatcherAgent(BaseAgent):
    """
    AI 简历分析师——告诉你"为什么简历和这个职位不匹配"。

    使用示例：
      agent = ResumeMatcherAgent()
      report = agent.analyze(resume_text, jd_text)
      print(report)
      # 输出：
      # ## Match Score: 82%
      # ## Matching Keywords: Python, Docker, FastAPI...
      # ## Missing Keywords: OpenAI(你简历里没写!), Kubernetes...
      # ## Improvement Suggestions:
      # 1. 在技能栏加上 OpenAI API
      # 2. 项目描述里强调 Docker 部署经验
    """

    config_class = ResumeMatcherConfig

    def __init__(self, config: ResumeMatcherConfig | None = None):
        super().__init__(config)
        self.matcher = ResumeMatcher(self.config, self.llm)

    # ============================================================
    #  三步 Pipeline
    # ============================================================

    def preprocess(self, inputs: dict, **kwargs) -> dict[str, Any]:
        """
        接收简历和职位描述。

        输入格式：{"resume": "张三的简历...", "jd": "招聘 Python 工程师..."}
        输出格式：{"resume_text": "张三的简历...", "jd_text": "招聘 Python 工程师..."}
        """
        return {
            "resume_text": inputs["resume"],
            "jd_text": inputs["jd"],
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        核心步骤：把简历和 JD 一起发给 AI，让它分析匹配度。

        AI 会根据精心设计的 system prompt（在 config.py 里定义）：
          1. 提取 JD 中的关键要求
          2. 在简历中找匹配的技能
          3. 计算匹配度
          4. 列出缺失项
          5. 给出改进建议
        """
        result = self.matcher.match(
            resume_text=model_inputs["resume_text"],
            jd_text=model_inputs["jd_text"],
        )
        return {"report": result.raw_report}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """AI 的回复已经是格式良好的 Markdown，直接返回。"""
        return model_outputs["report"]

    # ============================================================
    #  高级方法
    # ============================================================

    def analyze(self, resume_text: str, jd_text: str) -> str:
        """
        一键分析：给简历和 JD，拿回完整匹配报告。

        这是最常用的调用方式——一行代码完成分析。
        """
        return self.run({"resume": resume_text, "jd": jd_text})
