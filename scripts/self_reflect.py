"""自我反思工具 — Claude Code 调用，每完成一个任务后运行。

将任务执行轨迹、反思结果和教训写入项目记忆系统。

Usage:
    uv run python scripts/self_reflect.py "任务描述" --status ok|partial|fail [--lesson "教训"]

记忆存入:
    memory/facts.json — 项目事实
    memory/lessons.json — 经验教训
    memory/JOURNAL.jsonl — 操作日志
"""

import json
import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
LESSONS_PATH = os.path.join(MEMORY_DIR, "lessons.json")
JOURNAL_PATH = os.path.join(MEMORY_DIR, "JOURNAL.jsonl")


def _load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return default if default is not None else []


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_lesson(lesson: str, context: str, success: bool):
    """Add a lesson to lessons.json"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    lessons = _load_json(LESSONS_PATH, [])
    lessons.append({
        "lesson": lesson,
        "context": context,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    })
    if len(lessons) > 100:
        lessons = lessons[-100:]
    _save_json(LESSONS_PATH, lessons)
    status = "OK" if success else "FAIL"
    print(f"📖 教训已保存 [{status}]: {lesson}")


def append_journal(entry: str, tags: list[str] = None):
    """Append a journal entry"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(),
        "tags": tags or [],
        "text": entry,
    }
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"📔 日志已记录: {entry[:60]}...")


def show_status():
    """Show current memory state"""
    lessons = _load_json(LESSONS_PATH, [])
    print(f"\n记忆状态:")
    print(f"  教训总数: {len(lessons)}")
    recent = lessons[-5:]
    for l in recent:
        tag = "✅" if l.get("success") else "⚠️"
        print(f"  {tag} {l['ts'][:16]} {l['lesson'][:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自我反思工具")
    parser.add_argument("task", help="任务描述")
    parser.add_argument("--status", choices=["ok", "partial", "fail"], default="ok", help="任务状态")
    parser.add_argument("--lesson", help="学到的教训")
    parser.add_argument("--tags", help="逗号分隔的标签")
    parser.add_argument("--show", action="store_true", help="显示当前记忆状态")

    args = parser.parse_args()

    success = args.status == "ok"

    # Save lesson if provided
    if args.lesson:
        add_lesson(args.lesson, args.task, success)

    # Journal the task completion
    status_emoji = {"ok": "✅", "partial": "⚠️", "fail": "❌"}
    append_journal(
        f"{status_emoji[args.status]} {args.task}"
        + (f" | 教训: {args.lesson}" if args.lesson else ""),
        tags=(args.tags.split(",") if args.tags else None),
    )

    if args.show:
        show_status()
