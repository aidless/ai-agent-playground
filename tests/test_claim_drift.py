"""The README and the CI workflow may not drift apart or grow result shields."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
FAILURES = (ROOT / "FAILURE_MODES.md").read_text(encoding="utf-8")

PYTEST_SELECTION = "python -m pytest scripts/e2e_test.py tests"
SKIP_PREFIXES = (
    ".venv/",
    ".git/",
    "memory/knowledge/",
    "data/papers/",
    "chroma_db/",
)


def test_problem_statement_is_at_most_three_lines():
    preamble = README.split("## ", 1)[0]
    lines = [line for line in preamble.splitlines() if line.strip() and not line.startswith("#")]
    assert 1 <= len(lines) <= 3


def test_ci_pytest_selection_is_explicit_and_documented():
    assert PYTEST_SELECTION in CI
    assert PYTEST_SELECTION in README
    assert "--ignore=" not in CI
    assert "Historical cloud demo" in README
    assert "does not publish a demo URL" in README


def test_collect_only_count_matches_readme():
    env = os.environ.copy()
    env.setdefault("DEEPSEEK_API_KEY", "sk-ci-placeholder")
    env.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "scripts/e2e_test.py",
            "tests",
            "--collect-only",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    matches = re.findall(r"(\d+) tests collected", proc.stdout)
    assert matches, proc.stdout
    assert f"<!-- ci-collect-only: {matches[-1]} -->" in README


def test_readme_has_no_result_shields():
    banned = (
        "img.shields.io/badge/security",
        "img.shields.io/badge/code",
        "img.shields.io/badge/stress",
        "14/14",
        "1000/1000",
        "90%",
    )
    for item in banned:
        assert item not in README, item


def _author_needles() -> tuple[str, ...]:
    # Built at runtime so this file does not contain the path it forbids.
    user = "Admin" + "istrator"
    return (f"Users\\{user}", f"Users/{user}", f"C--Users-{user}")


def test_no_public_ip_demo_urls_or_author_desktops():
    ip_url = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}")
    author_needles = _author_needles()
    allowed_suffixes = {".py", ".md", ".yml", ".yaml", ".json", ".html", ".toml", ".txt", ".sh"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_PREFIXES) or "/__pycache__/" in f"/{rel}":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in ip_url.finditer(text):
            host = match.group(0).split("://", 1)[1]
            assert host.startswith("127.") or host.startswith("0.0.0.0"), rel
        for needle in author_needles:
            assert needle not in text, rel


def _yaml_code(text: str) -> str:
    """Drop full-line comments so a historical note is not a live swallow."""
    kept = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def test_bandit_gate_is_not_swallowed():
    code = _yaml_code(CI)
    assert "|| echo" not in code
    assert "|| true" not in code
    assert "continue-on-error" not in code
    assert "-b .bandit-baseline.json" in CI
    mutation = (ROOT / ".github" / "workflows" / "mutation-control.yml").read_text(encoding="utf-8")
    assert "B602" in mutation
    assert "gate_swallow" in mutation


def test_bandit_baseline_count_is_documented():
    data = json.loads((ROOT / ".bandit-baseline.json").read_text(encoding="utf-8"))
    count = len(data["results"])
    assert f"<!-- bandit-baseline-results: {count} -->" in FAILURES
