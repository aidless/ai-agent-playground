"""
Demo: Run the Resume Matcher against Liu Zewen's profile and the 立邦 JD.

This is a concrete, real-world use case — the same agent you built
analyzing your own fit for a job you're actually applying for.

Outputs a Markdown report to reports/liuzewen_vs_nippon.md
"""

from pathlib import Path

from resume_matcher.agent import ResumeMatcherAgent
from resume_matcher.extractor import extract_resume_text

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def main():
    resume_path = DATA_DIR / "liuzewen_resume.md"
    jd_path = DATA_DIR / "nippon_jd.md"

    print("=" * 60)
    print("  立邦 AI Agent 开发工程师 — 简历匹配分析")
    print("=" * 60)
    print(f"  简历: {resume_path.name}")
    print(f"  职位: {jd_path.name}")
    print()

    resume_text = extract_resume_text(str(resume_path))
    jd_text = extract_resume_text(str(jd_path))

    print(f"  简历字数: {len(resume_text)}")
    print(f"  JD 字数: {len(jd_text)}")
    print()
    print("  正在调用 AI 分析...")
    print()

    agent = ResumeMatcherAgent()
    report = agent.analyze(resume_text, jd_text)

    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = REPORTS_DIR / "liuzewen_vs_nippon.md"
    header = (
        "# 立邦 AI Agent 开发工程师 — 简历匹配分析报告\n\n"
        f"**候选人**: 刘泽文 | **职位**: AI Agent 开发工程师 | **公司**: 立邦投资有限公司\n\n"
        "---\n\n"
    )
    output_path.write_text(header + report, encoding="utf-8")

    print(report)
    print()
    print(f"  报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
