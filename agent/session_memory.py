"""Multi-Turn Conversation Memory — persistent cross-message context.

Enables the agent to remember previous exchanges within a session.
Each session has a unique ID that persists across messages.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).resolve().parent.parent / "memory" / "sessions"


class SessionMemory:
    """Stores conversation history per session for multi-turn context.

    Usage:
        store = SessionMemory()
        sid = store.create_session()
        store.add(sid, "user", "What is Python?")
        store.add(sid, "assistant", "Python is a programming language...")
        history = store.get_context(sid)
    """

    def __init__(self, max_history: int = 20, ttl_hours: int = 24):
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.ttl_hours = ttl_hours
        self._sessions: dict[str, list[dict]] = {}
        self._load()

    def _load(self):
        for f in SESSION_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sid = f.stem
                self._sessions[sid] = data.get("messages", [])
            except Exception:
                pass

    def _save(self, sid: str):
        path = SESSION_DIR / f"{sid}.json"
        path.write_text(json.dumps({
            "session_id": sid,
            "messages": self._sessions.get(sid, []),
            "updated": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_session(self) -> str:
        sid = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        self._sessions[sid] = []
        return sid

    def add(self, sid: str, role: str, content: str):
        if sid not in self._sessions:
            self._sessions[sid] = []
        self._sessions[sid].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Trim old messages
        if len(self._sessions[sid]) > self.max_history:
            self._sessions[sid] = self._sessions[sid][-self.max_history:]
        self._save(sid)

    def get_context(self, sid: str, last_n: int = 6) -> list[dict]:
        """Get recent conversation context for injection into agent prompt."""
        messages = self._sessions.get(sid, [])
        if not messages:
            return []
        # Return last N exchanges as context
        recent = messages[-last_n:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_summary(self, sid: str) -> str:
        """Generate a brief summary of conversation history."""
        messages = self._sessions.get(sid, [])
        if not messages:
            return ""
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        topics = " → ".join(m[:60] for m in user_msgs[-5:])
        return f"[Previous conversation: {len(messages)} messages. Topics: {topics}]"

    def cleanup(self):
        """Remove expired sessions."""
        cutoff = time.time() - self.ttl_hours * 3600
        for sid in list(self._sessions.keys()):
            path = SESSION_DIR / f"{sid}.json"
            if path.exists() and path.stat().st_mtime < cutoff:
                path.unlink()
                del self._sessions[sid]
