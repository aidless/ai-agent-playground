"""Automated security audit — generates CISO-ready compliance report.

Runs: bandit static analysis, dependency audit, custom OWASP checks.
Output: security_audit_report.json
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT))
        return r.returncode, r.stdout + r.stderr
    except Exception as e:
        return -1, str(e)


def check_owasp():
    """Custom OWASP Top 10 for LLM Apps checks."""
    findings = []

    # A01: Broken Access Control
    for f in (PROJECT / "agent").rglob("*.py"):
        content = f.read_text(encoding="utf-8", errors="replace")
        if "API_KEY" in content and not "os.environ" in content and not "os.getenv" in content:
            findings.append({"severity": "critical", "rule": "A01", "file": str(f.relative_to(PROJECT)),
                           "finding": "Potential hardcoded API key"})
        if "password" in content.lower() and "=" in content and not "os." in content:
            findings.append({"severity": "high", "rule": "A01", "file": str(f.relative_to(PROJECT)),
                           "finding": "Password-like variable assignment"})

    # A02: Cryptographic Failures
    # A03: Injection
    for f in (PROJECT / "agent").rglob("*.py"):
        content = f.read_text(encoding="utf-8", errors="replace")
        if "subprocess.run" in content and ("shell=True" in content):
            findings.append({"severity": "critical", "rule": "A03", "file": str(f.relative_to(PROJECT)),
                           "finding": "subprocess with shell=True"})

    # A05: Security Misconfiguration
    for f in (PROJECT / "agent").rglob("*.py"):
        content = f.read_text(encoding="utf-8", errors="replace")
        if "allow_origins=[\"*\"" in content or "allow_origins=['*']" in content:
            findings.append({"severity": "medium", "rule": "A05", "file": str(f.relative_to(PROJECT)),
                           "finding": "CORS allows all origins"})

    # Check .env not in git
    env_path = PROJECT / ".env"
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8", errors="replace")
        if "sk-" in env_content:
            findings.append({"severity": "info", "rule": "A05", "file": ".env",
                           "finding": ".env contains API key — ensure .env is gitignored"})

    # Check .gitignore
    gitignore = PROJECT / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8", errors="replace")
        for pattern in [".env", "*.key", "*.pem", "credentials*"]:
            if pattern not in content:
                findings.append({"severity": "high", "rule": "A05", "file": ".gitignore",
                               "finding": f"Missing gitignore pattern: {pattern}"})

    return findings


SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9_\-]{20,}', "API Key (sk-*)", "critical"),
    (r'sk-ant-[a-zA-Z0-9_\-]{20,}', "Anthropic API Key", "critical"),
    (r'AIza[0-9A-Za-z_\-]{35}', "Google API Key", "critical"),
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key", "critical"),
    (r'eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+', "JWT Token", "high"),
    (r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*', "Bearer Token", "high"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Token", "critical"),
    (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth Token", "critical"),
    (r'-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----', "Private Key", "critical"),
    (r'xox[baprs]-[a-zA-Z0-9-]+', "Slack Token", "high"),
    (r'github_pat_[a-zA-Z0-9_]{36}', "GitHub PAT", "critical"),
]


def scan_secrets(root: Path):
    """TruffleHog-style regex secret scan across all project files."""
    import re as _re
    findings = []
    skip_ext = {".pyc", ".pyd", ".exe", ".dll", ".so", ".dylib", ".lock", ".jsonl", ".jpg", ".png", ".gif"}

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix in skip_ext or f.stat().st_size > 1_000_000:
            continue
        if any(p in f.parts for p in [".git", "__pycache__", ".venv", "node_modules", "chroma_db"]):
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for pattern, name, severity in SECRET_PATTERNS:
            for match in _re.finditer(pattern, content):
                findings.append({
                    "severity": severity,
                    "rule": "secret_scan",
                    "file": str(f.relative_to(root)),
                    "line": content[:match.start()].count("\n") + 1,
                    "finding": f"Potential secret: {name}",
                    "match": match.group()[:40] + "...",
                })

    return findings


def main():
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bandit": {},
        "owasp_custom": [],
        "dependencies": {},
        "summary": {},
    }

    # 1. Bandit static analysis
    print("Running bandit...")
    rc, output = run(["uv", "run", "bandit", "-r", "agent/", "ai_agent_playground/", "scripts/", "-ll", "-f", "json"])
    if rc == 0 or rc == 1:  # 1 = findings but no errors
        try:
            # Bandit outputs JSON lines
            results["bandit"] = json.loads(output) if output.strip().startswith("{") else {"raw": output[:500]}
        except json.JSONDecodeError:
            results["bandit"] = {"raw": output[:1000]}
    else:
        results["bandit"] = {"error": output[:500]}

    # 2. OWASP custom checks
    print("Running OWASP checks...")
    results["owasp_custom"] = check_owasp()

    # 3. Secret scan
    print("Running secret scan...")
    results["secret_scan"] = scan_secrets(PROJECT)

    # 4. Dependency audit
    print("Running pip-audit...")
    rc, output = run(["uv", "run", "pip-audit", "--format", "json"], timeout=120)
    if rc == 0:
        try:
            results["dependencies"] = json.loads(output)
        except json.JSONDecodeError:
            results["dependencies"] = {"status": "clean", "raw": output[:500]}
    else:
        results["dependencies"] = {"raw": output[:500], "exit_code": rc}

    # 5. Summary
    bandit_issues = len(results["bandit"].get("results", [])) if isinstance(results["bandit"], dict) else "unknown"
    owasp_critical = sum(1 for f in results["owasp_custom"] if f["severity"] == "critical")
    owasp_high = sum(1 for f in results["owasp_custom"] if f["severity"] == "high")
    secret_critical = sum(1 for f in results["secret_scan"] if f["severity"] == "critical")
    secret_high = sum(1 for f in results["secret_scan"] if f["severity"] == "high")
    dep_vulns = len(results["dependencies"].get("dependencies", [])) if isinstance(results["dependencies"], dict) else "unknown"

    results["summary"] = {
        "bandit_findings": bandit_issues,
        "owasp_critical": owasp_critical,
        "owasp_high": owasp_high,
        "secret_critical": secret_critical,
        "secret_high": secret_high,
        "dependency_vulnerabilities": dep_vulns,
        "assessment": "CISO review required: automated checks completed",
        "recommendation": "Schedule third-party penetration test for production deployment",
    }

    report_path = PROJECT / "security_audit_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SECURITY AUDIT COMPLETE ===")
    print(f"Bandit: {bandit_issues} findings")
    print(f"OWASP custom: {owasp_critical} critical, {owasp_high} high")
    print(f"Secret scan: {secret_critical} critical, {secret_high} high")
    print(f"Dependencies: {dep_vulns} vulnerabilities")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
