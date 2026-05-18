"""自动记忆系统 — 解决"没有长时记忆"

每次工具调用自动记录，无需刻意保存。
跨会话记忆通过 memory/auto/ 目录持久化。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

MEMORY_AUTO_DIR = Path(__file__).resolve().parent.parent / "memory" / "auto"


class AutoMemory:
    """自动记忆——每次操作自动记录"""

    def __init__(self):
        MEMORY_AUTO_DIR.mkdir(parents=True, exist_ok=True)

    def record_action(self, action: str, target: str, result_summary: str, success: bool = True):
        """记录一次操作"""
        entry = {
            "ts": datetime.now().isoformat(),
            "action": action,
            "target": target,
            "result": result_summary[:200],
            "success": success,
        }
        path = MEMORY_AUTO_DIR / f"actions-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record_lesson(self, lesson: str, context: str = ""):
        """学到的教训——自动写入"""
        path = MEMORY_AUTO_DIR / "lessons.jsonl"
        entry = {
            "ts": datetime.now().isoformat(),
            "lesson": lesson,
            "context": context[:200],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent_actions(self, n: int = 10) -> list[dict]:
        """获取最近操作"""
        today = datetime.now().strftime("%Y-%m-%d")
        path = MEMORY_AUTO_DIR / f"actions-{today}.jsonl"
        if not path.exists():
            return []
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results[-n:]

    def get_recent_lessons(self, n: int = 5) -> list[dict]:
        """获取最近教训"""
        path = MEMORY_AUTO_DIR / "lessons.jsonl"
        if not path.exists():
            return []
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results[-n:]


# 全局单例
_auto = None


def get_auto_memory() -> AutoMemory:
    global _auto
    if _auto is None:
        _auto = AutoMemory()
    return _auto
