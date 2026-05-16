"""
Prompt Registry — versioned, searchable, diffable prompt management.

Why "change a prompt" is dangerous: you have no idea if v2 is better than v1.
The registry solves this by treating prompts like code — versioned, tracked, testable.

Usage:
    registry = PromptRegistry()
    registry.register("mcp_agent", "v1", "You are a helpful assistant...")
    registry.register("mcp_agent", "v2", "You are an expert tool-using AI...")

    latest = registry.get_latest("mcp_agent")
    diff = registry.diff("mcp_agent", "v1", "v2")
"""

import difflib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    """One version of a prompt."""

    name: str
    version: str
    content: str
    created_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def format(self, **kwargs) -> str:
        """Interpolate variables into the prompt template.

        Uses {variable_name} syntax compatible with Python's str.format().
        """
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing variable {e} in prompt '{self.name}' v{self.version}. "
                f"Available: {self._extract_variables()}"
            )

    def _extract_variables(self) -> list[str]:
        """Extract {variable} names from the template."""
        import re
        return re.findall(r'\{(\w+)\}', self.content)


class PromptRegistry:
    """Versioned prompt store — like git for prompts."""

    def __init__(self, storage_path: str | None = None):
        self._prompts: dict[str, list[PromptTemplate]] = {}
        self._storage_path = (
            Path(storage_path) if storage_path else None
        )
        if self._storage_path and self._storage_path.exists():
            self._load()

    # ---- Registration ----

    def register(
        self,
        name: str,
        version: str,
        content: str,
        meta: dict | None = None,
    ) -> PromptTemplate:
        """Register a new prompt version. Overwrites if same version exists."""
        template = PromptTemplate(
            name=name, version=version, content=content,
            meta=meta or {},
        )
        if name not in self._prompts:
            self._prompts[name] = []
        # Replace same version if exists
        existing = [p for p in self._prompts[name] if p.version == version]
        if existing:
            self._prompts[name].remove(existing[0])
        self._prompts[name].append(template)
        # Sort by creation time
        self._prompts[name].sort(key=lambda p: p.created_at)
        return template

    # ---- Retrieval ----

    def get(self, name: str, version: str) -> PromptTemplate | None:
        """Get a specific prompt version."""
        for p in self._prompts.get(name, []):
            if p.version == version:
                return p
        return None

    def get_latest(self, name: str) -> PromptTemplate | None:
        """Get the most recent version of a prompt."""
        prompts = self._prompts.get(name, [])
        return prompts[-1] if prompts else None

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a prompt."""
        return [p.version for p in self._prompts.get(name, [])]

    def list_all(self) -> list[str]:
        """List all registered prompt names."""
        return sorted(self._prompts.keys())

    # ---- Diff ----

    def diff(self, name: str, v1: str, v2: str) -> str:
        """Generate a unified diff between two prompt versions."""
        p1 = self.get(name, v1)
        p2 = self.get(name, v2)
        if not p1 or not p2:
            return f"Cannot diff: {name} ({v1}→{v2}) — version not found"

        diff_lines = difflib.unified_diff(
            p1.content.splitlines(keepends=True),
            p2.content.splitlines(keepends=True),
            fromfile=f"{name}:{v1}",
            tofile=f"{name}:{v2}",
            lineterm="",
        )
        return "".join(diff_lines)

    # ---- Export / Import ----

    def export_json(self, name: str | None = None) -> str:
        """Export prompts as JSON string."""
        prompts = (
            {name: self._prompts[name]} if name
            else self._prompts
        )
        data = {}
        for n, versions in prompts.items():
            data[n] = [
                {
                    "version": p.version,
                    "content": p.content,
                    "created_at": p.created_at,
                    "meta": p.meta,
                }
                for p in versions
            ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def save(self):
        """Persist to disk."""
        if self._storage_path:
            self._storage_path.write_text(
                self.export_json(), encoding="utf-8"
            )

    def _load(self):
        """Load from disk."""
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        for name, versions in data.items():
            for v in versions:
                self.register(
                    name=name,
                    version=v["version"],
                    content=v["content"],
                    meta=v.get("meta", {}),
                )


# ============================================================
#  Pre-built prompt library
# ============================================================


def create_default_registry() -> PromptRegistry:
    """Create a registry pre-loaded with project prompts for versioning."""
    r = PromptRegistry()

    # MCP Agent — v1 (original) vs v2 (enhanced)
    r.register(
        "mcp_agent", "v1",
        "You are an AI assistant with access to tools. "
        "Use tools when you need real-time information, file access, or computation.\n\n"
        "Tool use format:\n"
        "When you need to use a tool, respond with:\n"
        '{"tool": "tool_name", "args": {"arg1": "value1"}}\n\n'
        "After receiving tool results, continue with your answer.\n"
        "Only use tools when necessary. For general questions, answer directly.",
        meta={"author": "liuzewen", "description": "Original generic assistant prompt"},
    )
    r.register(
        "mcp_agent", "v2",
        "You are an expert AI agent with tool-use capabilities. "
        "Analyze each user request and determine if tools are needed.\n\n"
        "Decision framework:\n"
        "1. For calculations → use calculator tool\n"
        "2. For file operations → use read_file or write_file\n"
        "3. For up-to-date info → use web_search\n"
        "4. For general knowledge questions → answer directly WITHOUT tools\n\n"
        "Tool call format (JSON only, no extra text):\n"
        '{"tool": "tool_name", "args": {"arg1": "value1"}}\n\n'
        "After receiving tool results, synthesize a clear, concise answer. "
        "Cite tool results when relevant. Never invent data — use tools instead.",
        meta={"author": "liuzewen", "description": "Enhanced with explicit decision framework"},
    )

    # Resume Matcher — v1 (English) vs v2 (Chinese JD support)
    r.register(
        "resume_matcher", "v1",
        "You are an expert resume analyst and career coach. "
        "Your task is to compare a candidate's resume against a job description "
        "and provide actionable feedback.\n\n"
        "Analyze the following:\n"
        "1. Match Score: overall percentage match (0-100%)\n"
        "2. Matching Keywords: skills/qualifications found in both resume and JD\n"
        "3. Missing Keywords: important JD requirements NOT in the resume\n"
        "4. Resume Gaps: specific sections or experiences the candidate should add\n"
        "5. Improvement Suggestions: 3-5 concrete bullet points to improve match",
        meta={"author": "liuzewen", "description": "Original English-only prompt"},
    )
    r.register(
        "resume_matcher", "v2",
        "You are an expert AI recruitment analyst specializing in AI Agent / LLM "
        "application development positions. Support both Chinese and English.\n\n"
        "Score across 5 independent dimensions (each 0-100%):\n"
        "1. Tech Stack Match\n"
        "2. Project Experience Match\n"
        "3. Theoretical Foundation Match\n"
        "4. Soft Skills Match\n"
        "5. Overall Match (weighted holistic)\n\n"
        "Also predict 3-5 interview questions based on gaps found. "
        "Be ruthlessly honest — candidates need real feedback.",
        meta={"author": "liuzewen", "description": "5-dimensional scoring + interview predictions"},
    )

    return r
