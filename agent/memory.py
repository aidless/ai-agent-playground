"""Agent 持久化记忆系统

让 Agent 拥有跨会话的"记忆力"：
- 事实记忆（fact）：明确的、静态的知识（如"用户叫泽文"）
- 教训记忆（lesson）：从经验中总结的教训（如"调用 X 工具前需要先做 Y"）
- 痕迹记忆（trace）：每次执行轨迹，用于事后分析

记忆存储在 project/memory/ 目录下，以 JSON 文件形式持久化。
"""

import json
import os
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 记忆存储路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
FACTS_PATH = os.path.join(MEMORY_DIR, "facts.json")
LESSONS_PATH = os.path.join(MEMORY_DIR, "lessons.json")
TRACES_DIR = os.path.join(MEMORY_DIR, "traces")


def _ensure_dirs():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(TRACES_DIR, exist_ok=True)


def _load_json(path: str, default: Any = None) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("记忆文件 %s 损坏，重置: %s", path, e)
    return default if default is not None else {}


def _save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class AgentMemory:
    """Agent 记忆系统 — 线程安全（单线程 Agent 场景）"""

    def __init__(self):
        _ensure_dirs()
        self.facts: dict = _load_json(FACTS_PATH, {})
        self.lessons: list = _load_json(LESSONS_PATH, [])

    # ── 事实记忆 ──────────────────────────────────

    def save_fact(self, key: str, value: str, source: str = ""):
        """记住一个事实"""
        self.facts[key] = {
            "value": value,
            "source": source,
            "updated_at": datetime.now().isoformat(),
        }
        _save_json(FACTS_PATH, self.facts)
        logger.info("🧠 记住事实: %s = %s", key, value)

    def recall_fact(self, key: str) -> Optional[str]:
        """回忆一个事实"""
        entry = self.facts.get(key)
        return entry["value"] if entry else None

    def search_facts(self, query: str) -> list[dict]:
        """搜索相关事实（关键词匹配）"""
        q = query.lower()
        results = []
        for key, data in self.facts.items():
            if q in key.lower() or q in data["value"].lower():
                results.append({"key": key, **data})
        return results

    def forget_fact(self, key: str):
        """遗忘一个事实"""
        self.facts.pop(key, None)
        _save_json(FACTS_PATH, self.facts)

    # ── 教训记忆 ──────────────────────────────────

    def add_lesson(self, lesson: str, context: str = "", success: bool = False):
        """从经验中学习一条教训"""
        self.lessons.append({
            "lesson": lesson,
            "context": context,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        # 只保留最近 100 条
        if len(self.lessons) > 100:
            self.lessons = self.lessons[-100:]
        _save_json(LESSONS_PATH, self.lessons)
        tag = "✅" if success else "⚠️"
        logger.info(f"{tag} 学到教训: {lesson}")

    def get_recent_lessons(self, n: int = 5) -> list[dict]:
        """获取最近的教训"""
        return self.lessons[-n:]

    def search_lessons(self, query: str) -> list[dict]:
        """搜索相关教训"""
        q = query.lower()
        return [l for l in self.lessons if q in l["lesson"].lower() or q in l["context"].lower()]

    # ── 痕迹记忆 ──────────────────────────────────

    def save_trace(self, trace_id: str, steps: list[dict]):
        """保存一次执行轨迹"""
        path = os.path.join(TRACES_DIR, f"{trace_id}.json")
        payload = {
            "trace_id": trace_id,
            "completed_at": datetime.now().isoformat(),
            "steps": steps,
        }
        _save_json(path, payload)

    def load_trace(self, trace_id: str) -> Optional[dict]:
        """加载一次执行轨迹"""
        path = os.path.join(TRACES_DIR, f"{trace_id}.json")
        return _load_json(path)

    def list_traces(self, n: int = 10) -> list[str]:
        """列出最近的轨迹 ID"""
        if not os.path.isdir(TRACES_DIR):
            return []
        files = sorted(os.listdir(TRACES_DIR), reverse=True)[:n]
        return [f.replace(".json", "") for f in files if f.endswith(".json")]

    # ── 自我认知 ──────────────────────────────────

    def summarize_identity(self) -> str:
        """生成 Agent 的自我认知摘要（注入 system prompt 用）"""
        parts = ["## 我的记忆"]
        for key, data in self.facts.items():
            parts.append(f"- {key}: {data['value']}")
        recent = self.get_recent_lessons(3)
        if recent:
            parts.append("\n## 我最近学到的教训")
            for l in recent:
                tag = "✅" if l.get("success") else "⚠️"
                parts.append(f"- {tag} {l['lesson']}")
        return "\n".join(parts)


# 全局单例
_memory: Optional[AgentMemory] = None


def get_memory() -> AgentMemory:
    global _memory
    if _memory is None:
        _memory = AgentMemory()
    return _memory


def reset_memory():
    global _memory
    _memory = None
