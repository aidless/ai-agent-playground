"""技能系统 — 兼容 agentskills.io 标准（Hermes Agent 式自创建 + Anthropic 式格式）

完成复杂任务后，Agent 自动将经验封装为可复用技能：
    1. 从任务执行轨迹中提取模式
    2. 生成 SKILL.md（YAML frontmatter + markdown 指令）
    3. 存入 skills/ 目录
    4. 下次类似任务自动加载相关技能

技能格式遵循 agentskills.io 标准:
    ---
    name: skill-name
    description: When and why to use this skill
    ---
    # Instructions (markdown)
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _ensure_dir():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 YAML frontmatter，返回 (metadata, body)

    优先使用 yaml 库，回退到简单 key:value 模式。
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_str = parts[1].strip()
    body = parts[2].strip()

    metadata = {}

    # 尝试 yaml 库
    try:
        import yaml
        metadata = yaml.safe_load(frontmatter_str) or {}
        return metadata, body
    except ImportError:
        pass
    except Exception:
        pass

    # 回退：简单 key:value 解析
    for line in frontmatter_str.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            metadata[key] = value

    return metadata, body


def build_frontmatter(name: str, description: str, **extra) -> str:
    """构建 YAML frontmatter 字符串"""
    lines = ["---", f"name: {name}", f"description: {description}"]
    for k, v in extra.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


class Skill:
    """单个技能实例"""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.metadata: dict = {}
        self.body: str = ""
        self._load()

    def _load(self):
        if self.path.exists():
            content = self.path.read_text(encoding="utf-8")
            self.metadata, self.body = parse_frontmatter(content)
            self.name = self.metadata.get("name", self.name)

    @property
    def description(self) -> str:
        return self.metadata.get("description", "")


class SkillManager:
    """技能管理器 — 创建、搜索、加载技能

    用法:
        manager = SkillManager()
        manager.create_from_experience("编写 FastAPI 中间件", trace, success=True)
        best = manager.search("FastAPI middleware")
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or SKILLS_DIR
        _ensure_dir()

    def create(
        self,
        name: str,
        description: str,
        body: str,
        metadata: Optional[dict] = None,
    ) -> Skill:
        """显式创建一个技能"""
        _ensure_dir()

        frontmatter = build_frontmatter(name, description, **(metadata or {}))
        content = f"{frontmatter}\n\n{body}\n"

        path = self.skills_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")

        logger.info("Skill created: %s", name)
        return Skill(path)

    def create_from_experience(
        self,
        task: str,
        trace: list[dict],
        success: bool = True,
        llm_client=None,
        model: str = "deepseek-chat",
    ) -> Optional[Skill]:
        """从任务执行经验中自动创建技能（Hermes Agent 风格）

        Args:
            task: 原始任务描述
            trace: 执行轨迹 steps
            success: 是否成功
            llm_client: LLM 客户端（用于生成技能内容）
            model: 模型名称

        Returns:
            创建的 Skill 实例，或 None
        """
        if not llm_client:
            logger.debug("No LLM client, skipping auto skill creation")
            return None

        if not success:
            logger.debug("Task failed, still extracting patterns...")

        # 从轨迹中提取关键步骤
        key_steps = []
        for step in trace:
            event = step.get("event", "")
            if event in ("tool_call", "planning", "reflect_result", "done"):
                key_steps.append(step)

        trace_summary = json.dumps(key_steps, ensure_ascii=False, indent=2)[:2000]

        # 让 LLM 生成技能
        prompt = f"""Based on this task execution, create a reusable skill.

Task: {task}

Execution trace:
{trace_summary}

Generate a skill in this format:
NAME: a-short-hyphenated-name
DESCRIPTION: one sentence describing when to use this skill
BODY: markdown instructions for the next agent (what pattern to follow, what pitfalls to avoid, what tools to use in what order). Keep it to 3-8 bullet points.

Respond with:
---NAME---
<name>
---DESCRIPTION---
<description>
---BODY---
<body>"""

        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            logger.warning("LLM call failed during skill creation: %s", e)
            return None

        # 解析 LLM 输出
        name_match = re.search(r"---NAME---\s*\n(.+?)(?:\n---|\Z)", raw, re.DOTALL)
        desc_match = re.search(r"---DESCRIPTION---\s*\n(.+?)(?:\n---|\Z)", raw, re.DOTALL)
        body_match = re.search(r"---BODY---\s*\n(.+)", raw, re.DOTALL)

        if not (name_match and body_match):
            logger.warning("Failed to parse skill from LLM output")
            return None

        skill_name = name_match.group(1).strip().replace(" ", "-").lower()
        skill_desc = desc_match.group(1).strip() if desc_match else task[:100]
        skill_body = body_match.group(1).strip()

        return self.create(
            name=skill_name,
            description=skill_desc,
            body=skill_body,
            metadata={
                "source": "auto-generated",
                "created_at": datetime.now().isoformat(),
                "source_task": task[:200],
                "success": success,
            },
        )

    def search(self, query: str) -> list[Skill]:
        """搜索相关技能（分词匹配 name + description + body）"""
        _ensure_dir()
        # 提取有意义的词（>=3 字符）
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z\u4e00-\u9fff_\-]{3,}", query)]

        scored: list[tuple[int, Skill]] = []
        for path in self.skills_dir.glob("*.md"):
            try:
                skill = Skill(path)
            except Exception:
                continue

            score = 0
            text = (skill.name + " " + skill.description + " " + skill.body[:500]).lower()
            for t in tokens:
                if t in text:
                    score += 1
                if t in skill.name.lower():
                    score += 2  # 名称匹配权重更高
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:10]]

    def list_all(self) -> list[Skill]:
        """列出所有技能"""
        _ensure_dir()
        skills = []
        for path in sorted(self.skills_dir.glob("*.md")):
            try:
                skills.append(Skill(path))
            except Exception:
                continue
        return skills

    def get(self, name: str) -> Optional[Skill]:
        """获取指定技能"""
        path = self.skills_dir / f"{name}.md"
        if path.exists():
            return Skill(path)
        path = self.skills_dir / f"{name}"
        if path.exists():
            return Skill(path)
        return None

    def inject_context(self, task: str, max_skills: int = 3) -> str:
        """为当前任务注入相关技能上下文（给 LLM 用）"""
        relevant = self.search(task)[:max_skills]
        if not relevant:
            return ""

        parts = ["[Available Skills — follow these if relevant]"]
        for s in relevant:
            parts.append(f"\n## {s.name}\n{s.description}\n\n{s.body[:500]}")
        return "\n".join(parts)
