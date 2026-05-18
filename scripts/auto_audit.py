"""Project Auto Audit — 项目自动健康检查

Claude Code 自主调用，扫描项目状态、发现潜在问题。

Usage:
    uv run python scripts/auto_audit.py

Output: JSON with findings, suggestions, and project metrics
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: str | None = None) -> str:
    """Run a command and return stdout."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def check_git_status():
    """Check for uncommitted changes, branch info."""
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    changes = [line for line in status.split("\n") if line.strip()] if status else []
    ahead = run(["git", "rev-list", "--count", f"origin/{branch}..HEAD", "--"]) if branch != "HEAD" else "?"
    return {
        "branch": branch,
        "uncommitted_files": len(changes),
        "uncommitted_detail": changes[:10],  # first 10
        "commits_ahead": ahead,
    }


def check_deps_outdated():
    """Check if uv.lock is out of sync with pyproject.toml."""
    r = run(["uv", "lock", "--check"])
    return {"lock_synced": "up to date" in r.lower() or "fresh" in r.lower(), "detail": r[:200]}


def check_unused_files():
    """Find potential dead files."""
    dead_patterns = [
        "**/traces.jsonl", "**/*.tmp", "check_*.py", "**/__pycache__",
    ]
    findings = []
    for pattern in dead_patterns:
        for f in PROJECT_ROOT.glob(pattern):
            findings.append(str(f.relative_to(PROJECT_ROOT)))
    return {"dead_files": findings}


def check_test_health():
    """Run pytest --collect-only to find tests."""
    r = run(["uv", "run", "python", "-m", "pytest", "--collect-only", "-q", "--no-header"])
    # Count test lines
    lines = [l for l in (r.split("\n") if r else []) if l.strip() and not l.startswith("<") and "no tests ran" not in l]
    test_count = len([l for l in lines if l.startswith("  ") or "." in l[:5]])
    errors = [l for l in lines if "ERROR" in l.upper()]
    return {"test_count": test_count or "?", "errors": errors[:5], "raw": r[:500]}


def check_env():
    """Check critical env vars."""
    env_file = PROJECT_ROOT / ".env"
    vars_ok = {
        "has_env_file": env_file.exists(),
        "deepseek_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "gateway_key_set": bool(os.environ.get("GATEWAY_API_KEY")),
    }
    return vars_ok


def check_disk_usage():
    """Check memory dir size."""
    mem_dir = PROJECT_ROOT / "memory"
    if mem_dir.exists():
        total = sum(f.stat().st_size for f in mem_dir.rglob("*") if f.is_file())
        return {"memory_dir_size_kb": total // 1024}
    return {"memory_dir_size_kb": 0}


def check_entry_points():
    """Verify key files exist and are non-empty."""
    key_files = [
        "agent/server.py",
        "agent/async_core.py",
        "agent/memory.py",
        "agent/tools/registry.py",
        "agent/tools/web_search.py",
        "agent/tools/code_exec.py",
        "scripts/search_web.py",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
    ]
    results = {}
    for f in key_files:
        p = PROJECT_ROOT / f
        results[f] = p.exists() and p.stat().st_size > 0
    return {"key_files": results}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from datetime import datetime
    report = {
        "timestamp": datetime.now().isoformat(),
        "git": check_git_status(),
        "deps": check_deps_outdated(),
        "files": check_unused_files(),
        "tests": check_test_health(),
        "env": check_env(),
        "disk": check_disk_usage(),
        "entry_points": check_entry_points(),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
