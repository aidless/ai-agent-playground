"""
State Manager — externalized memory for long-running agents.

The "naked model trap": LLMs running long tasks inevitably:
  - Try to do everything at once → exhaust context window
  - See partial progress and declare victory → no verification
  - Forget what they did 10 steps ago → repeat or contradict

Solution (from Anthropic's harness engineering post): external artifacts as memory.
  - Task checklist: JSON manifest tracking what's done / in-progress / pending
  - Progress journal: step-by-step log written to disk (survives context loss)
  - Git checkpoints: snapshot before risky operations, rollback on failure
  - Resume: on restart, read journal → rebuild context → continue

This is NOT "prompt engineering" — it's infrastructure that constrains the agent
to work reliably across long time horizons.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
#  Task Checkpoint — track what needs doing
# ============================================================


@dataclass
class TaskItem:
    """One item in the agent's task checklist."""

    id: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | failed | skipped
    result: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class TaskManifest:
    """The agent's working memory — persisted to JSON, read on restart."""

    session_id: str
    goal: str
    created_at: str
    items: list[TaskItem] = field(default_factory=list)
    git_checkpoints: list[str] = field(default_factory=list)
    journal_entries: list[str] = field(default_factory=list)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.items if t.status == "completed")

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def progress_pct(self) -> float:
        return self.completed_count / self.total_count if self.total_count else 0.0

    @property
    def failed_items(self) -> list[TaskItem]:
        return [t for t in self.items if t.status == "failed"]

    @property
    def next_pending(self) -> TaskItem | None:
        for t in self.items:
            if t.status == "pending":
                return t
        return None


class StateManager:
    """Manages agent state across tool calls, restarts, and failures.

    The core insight from Anthropic: "use external artifacts as memory."
    The agent doesn't need to remember everything in its context window —
    it reads state from disk, acts, writes state back.

    Usage:
        sm = StateManager(work_dir="./agent_workspace")
        sm.start_session("Build a REST API for user management")

        sm.add_task("create_models", "Define User and Role models")
        sm.start_task("create_models")
        # ... agent does the work ...
        sm.complete_task("create_models", "Created User and Role in models.py")

        sm.checkpoint("models_created")  # Git snapshot

        # On crash/restart:
        sm.resume()  # Reads manifest, rebuilds context from journal
    """

    def __init__(self, work_dir: str = "./agent_workspace"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.work_dir / "manifest.json"
        self.journal_path = self.work_dir / "journal.md"
        self._manifest: TaskManifest | None = None
        self._is_git = (self.work_dir / ".git").exists()

    # ============================================================
    #  Session lifecycle
    # ============================================================

    def start_session(self, goal: str) -> TaskManifest:
        """Start a new agent session. Creates manifest + journal."""
        import uuid

        self._manifest = TaskManifest(
            session_id=str(uuid.uuid4())[:8],
            goal=goal,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save()
        self._write_journal(
            f"# Session {self._manifest.session_id}\n\n"
            f"**Goal**: {goal}\n\n"
            f"**Started**: {self._manifest.created_at}\n\n"
            f"## Progress Log\n"
        )
        return self._manifest

    def resume(self) -> TaskManifest | None:
        """Resume from a previous session. Rebuilds context from journal."""
        if not self.manifest_path.exists():
            return None

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._manifest = TaskManifest(
            session_id=data["session_id"],
            goal=data["goal"],
            created_at=data["created_at"],
            items=[
                TaskItem(
                    id=t["id"], description=t["description"],
                    status=t["status"], result=t.get("result", ""),
                    error=t.get("error", ""),
                    started_at=t.get("started_at", ""),
                    completed_at=t.get("completed_at", ""),
                )
                for t in data.get("items", [])
            ],
            git_checkpoints=data.get("git_checkpoints", []),
            journal_entries=data.get("journal_entries", []),
        )

        # Read journal to provide context
        journal = ""
        if self.journal_path.exists():
            journal = self.journal_path.read_text(encoding="utf-8")

        self._write_journal(
            f"\n\n### [Resumed at {datetime.now(timezone.utc).isoformat()}]\n"
        )

        print(f"[StateManager] Resumed session {self._manifest.session_id}")
        print(f"  Goal: {self._manifest.goal}")
        print(f"  Progress: {self._manifest.completed_count}/{self._manifest.total_count}")
        print(f"  Next: {self._manifest.next_pending.description if self._manifest.next_pending else 'All done!'}")

        return self._manifest

    @property
    def manifest(self) -> TaskManifest:
        if self._manifest is None:
            raise RuntimeError("No active session. Call start_session() or resume() first.")
        return self._manifest

    # ============================================================
    #  Task operations
    # ============================================================

    def add_task(self, task_id: str, description: str):
        """Add a task to the checklist."""
        self.manifest.items.append(TaskItem(id=task_id, description=description))
        self._save()
        self._log(f"Task added: [{task_id}] {description}")

    def add_tasks(self, tasks: list[tuple[str, str]]):
        """Batch-add tasks. tasks = [(id, description), ...]"""
        for tid, desc in tasks:
            self.add_task(tid, desc)

    def start_task(self, task_id: str):
        """Mark a task as in-progress."""
        task = self._find(task_id)
        task.status = "in_progress"
        task.started_at = datetime.now(timezone.utc).isoformat()
        self._save()
        self._log(f"Started: [{task_id}] {task.description}")

    def complete_task(self, task_id: str, result: str = ""):
        """Mark a task as completed with result."""
        task = self._find(task_id)
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now(timezone.utc).isoformat()
        self._save()
        self._log(f"Completed: [{task_id}] {task.description} — {result[:200]}")

    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed with error message."""
        task = self._find(task_id)
        task.status = "failed"
        task.error = error
        task.completed_at = datetime.now(timezone.utc).isoformat()
        self._save()
        self._log(f"FAILED: [{task_id}] {task.description} — {error[:200]}")

    def skip_task(self, task_id: str, reason: str = ""):
        """Skip a task (not needed)."""
        task = self._find(task_id)
        task.status = "skipped"
        task.result = reason
        self._save()

    # ============================================================
    #  Git checkpointing
    # ============================================================

    def checkpoint(self, label: str):
        """Create a Git checkpoint before a risky operation.

        If something goes wrong, the agent can rollback to this point.
        """
        if not self._is_git:
            self.manifest.git_checkpoints.append(
                f"{datetime.now(timezone.utc).isoformat()} | {label} | [no git]"
            )
            self._save()
            self._log(f"[no git] Checkpoint: {label}")
            return

        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self.work_dir), capture_output=True, text=True,
                timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", f"checkpoint: {label}"],
                cwd=str(self.work_dir), capture_output=True, text=True,
                timeout=10,
            )
            commit_hash = ""
            if result.returncode == 0:
                # Extract commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(self.work_dir), capture_output=True, text=True,
                    timeout=5,
                )
                commit_hash = hash_result.stdout.strip()[:8]

            self.manifest.git_checkpoints.append(
                f"{datetime.now(timezone.utc).isoformat()} | {label} | {commit_hash}"
            )
            self._save()
            self._log(f"Checkpoint: {label} ({commit_hash})")
        except Exception as e:
            self._log(f"Checkpoint failed: {e}")

    def rollback(self, steps: int = 1):
        """Roll back to a previous checkpoint."""
        if not self._is_git:
            self._log("[no git] Cannot rollback")
            return
        try:
            subprocess.run(
                ["git", "reset", "--hard", f"HEAD~{steps}"],
                cwd=str(self.work_dir), capture_output=True, text=True,
                timeout=10,
            )
            self._log(f"Rolled back {steps} checkpoint(s)")
        except Exception as e:
            self._log(f"Rollback failed: {e}")

    # ============================================================
    #  Context rebuild (for agent restart)
    # ============================================================

    def rebuild_context(self) -> str:
        """Rebuild a context summary for the agent after restart.

        This is what the agent reads to understand "where we left off" —
        no reliance on context window memory.

        Returns a formatted string suitable for injecting into the system prompt.
        """
        m = self.manifest
        parts = [
            f"## Session Context (Rebuilt from State)",
            f"Goal: {m.goal}",
            f"Session: {m.session_id}",
            f"Progress: {m.completed_count}/{m.total_count} tasks completed ({m.progress_pct:.0%})",
            "",
            "### Task Status",
        ]

        for item in m.items:
            icon = {"completed": "✅", "in_progress": "🔄", "failed": "❌",
                    "pending": "⬜", "skipped": "⏭️"}.get(item.status, "❓")
            parts.append(f"{icon} [{item.id}] {item.description}")
            if item.status == "failed":
                parts.append(f"   Error: {item.error}")
            if item.result:
                parts.append(f"   Result: {item.result[:200]}")

        if m.git_checkpoints:
            parts.append("\n### Checkpoints")
            for cp in m.git_checkpoints[-5:]:  # Last 5
                parts.append(f"- {cp}")

        # Include recent journal entries
        if m.journal_entries:
            parts.append("\n### Recent Activity")
            for entry in m.journal_entries[-10:]:  # Last 10 entries
                parts.append(f"- {entry}")

        return "\n".join(parts)

    # ============================================================
    #  Internals
    # ============================================================

    def _find(self, task_id: str) -> TaskItem:
        for t in self.manifest.items:
            if t.id == task_id:
                return t
        raise KeyError(f"Task not found: {task_id}")

    def _save(self):
        """Persist manifest to disk as JSON."""
        m = self.manifest
        data = {
            "session_id": m.session_id,
            "goal": m.goal,
            "created_at": m.created_at,
            "items": [
                {
                    "id": t.id, "description": t.description,
                    "status": t.status, "result": t.result,
                    "error": t.error,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                }
                for t in m.items
            ],
            "git_checkpoints": m.git_checkpoints,
            "journal_entries": m.journal_entries,
        }
        self.manifest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log(self, message: str):
        """Add a journal entry."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.manifest.journal_entries.append(entry)
        self._write_journal(f"{entry}\n")

    def _write_journal(self, text: str):
        """Append to the Markdown journal file."""
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(text)
