"""Sandboxed Meta Self-Evolution — HYPERAGENTS-style safe self-modification.

The MetaAgent can experiment on a COPY of its own code inside an isolated
sandbox. If the experiment passes all tests + safety checks, a proposal is
generated for human review. If it fails, the sandbox is destroyed.

This closes the final gap: the meta agent can safely improve itself without
risk of breaking the running system.

Architecture:
  1. Snapshot current agent code → copy to sandbox directory
  2. MetaAgent generates proposed changes via LLM
  3. Apply changes to sandbox copy
  4. Run test suite against sandbox copy
  5. If all tests pass + safety check → proposal saved
  6. If tests fail → sandbox destroyed, failure logged for learning
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SANDBOX_DIR = Path(__file__).resolve().parent.parent / "sandbox_workspace" / "meta_experiments"


@dataclass
class SandboxExperiment:
    experiment_id: str
    target_file: str
    proposed_changes: str = ""
    diff: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    safety_check: bool = False
    applied: bool = False
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


META_IMPROVEMENT_PROMPT = (
    "You are analyzing the agent's OWN source code to find improvements. "
    "Focus on: error handling, edge cases, performance optimizations, "
    "code clarity. Do NOT change the public API or remove safety checks.\n\n"
    "File: {file_path}\n"
    "Current code:\n```python\n{code}\n```\n\n"
    "Generate an improved version. If no meaningful improvement is possible, "
    "say 'NO_CHANGE'. Output ONLY the improved code or 'NO_CHANGE'."
)


class SandboxMetaEvolution:
    """Safe sandbox for MetaAgent self-experimentation.

    Usage:
        sandbox = SandboxMetaEvolution(project_root, agent)
        result = await sandbox.experiment("agent/meta_agent.py")
        if result.applied:
            print(f"Proposal ready: {result.experiment_id}")
    """

    def __init__(self, project_root: Path, agent):
        self.project_root = Path(project_root)
        self.agent = agent
        self._history: list[SandboxExperiment] = []
        SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    async def experiment(self, target_file: str) -> SandboxExperiment:
        """Run a safe self-modification experiment on a copy of the code."""
        import uuid
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        exp = SandboxExperiment(experiment_id=exp_id, target_file=target_file)
        sandbox_path = SANDBOX_DIR / exp_id

        try:
            # Step 1: Create sandbox copy
            logger.info("SandboxMeta: creating sandbox at %s", sandbox_path)
            self._create_sandbox(sandbox_path)

            # Step 2: Read target file
            original_path = self.project_root / target_file
            if not original_path.exists():
                exp.error = f"Target file not found: {target_file}"
                return exp
            original_code = original_path.read_text(encoding="utf-8")

            # Step 3: Generate improvement via LLM
            improvement = await self._generate_improvement(target_file, original_code)
            if not improvement or improvement.strip() == "NO_CHANGE":
                exp.error = "No improvement generated"
                exp.proposed_changes = "NO_CHANGE"
                self._cleanup(sandbox_path)
                return exp
            exp.proposed_changes = improvement

            # Step 4: Apply changes to sandbox copy
            sandbox_target = sandbox_path / target_file
            sandbox_target.parent.mkdir(parents=True, exist_ok=True)
            sandbox_target.write_text(improvement, encoding="utf-8")

            # Step 5: Compute diff
            import difflib
            exp.diff = "\n".join(difflib.unified_diff(
                original_code.splitlines(), improvement.splitlines(),
                fromfile=f"{target_file}.original", tofile=f"{target_file}.proposed",
                lineterm="",
            ))

            # Step 6: Safety check (AST + import scan)
            exp.safety_check = self._safety_check(improvement)
            if not exp.safety_check:
                exp.error = "Safety check failed"
                self._cleanup(sandbox_path)
                return exp

            # Step 7: Run tests in sandbox
            passed, failed = await self._run_tests_in_sandbox(sandbox_path)
            exp.tests_passed = passed
            exp.tests_failed = failed

            # Step 8: Decide
            if failed == 0 and passed > 0:
                # Save as proposal
                self._save_proposal(exp)
                exp.applied = True
                logger.info("SandboxMeta: experiment %s PASSED (%d tests)", exp_id, passed)
            else:
                exp.error = f"Tests: {passed} passed, {failed} failed"
                logger.warning("SandboxMeta: experiment %s FAILED", exp_id)

        except Exception as e:
            exp.error = str(e)
            logger.error("SandboxMeta: experiment %s crashed: %s", exp_id, e)
        finally:
            self._cleanup(sandbox_path)

        self._history.append(exp)
        if len(self._history) > 50:
            self._history = self._history[-25:]
        self._save_experiment_log(exp)
        return exp

    def _create_sandbox(self, sandbox_path: Path):
        """Copy agent/ directory to sandbox."""
        sandbox_path.mkdir(parents=True, exist_ok=True)
        agent_src = self.project_root / "agent"
        agent_dst = sandbox_path / "agent"
        if agent_src.exists():
            shutil.copytree(str(agent_src), str(agent_dst), dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Also copy tests
        tests_src = self.project_root / "tests"
        tests_dst = sandbox_path / "tests"
        if tests_src.exists():
            shutil.copytree(str(tests_src), str(tests_dst), dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Copy pyproject.toml
        pyproj = self.project_root / "pyproject.toml"
        if pyproj.exists():
            shutil.copy2(str(pyproj), str(sandbox_path / "pyproject.toml"))

    async def _generate_improvement(self, file_path: str, code: str) -> str:
        """Ask LLM to generate code improvement."""
        try:
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[
                    {"role": "system", "content": META_IMPROVEMENT_PROMPT.format(
                        file_path=file_path, code=code[:5000],
                    )},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
            if isinstance(text, str):
                text = text.strip()
            else:
                text = str(text).strip()

            # Clean markdown
            import re
            if text.startswith("```"):
                text = re.sub(r'^```(?:python)?\s*\n', '', text)
                text = re.sub(r'\n```\s*$', '', text)
            return text
        except Exception as e:
            logger.warning("SandboxMeta: LLM improvement failed: %s", e)
            return ""

    def _safety_check(self, code: str) -> bool:
        """AST safety check."""
        import ast
        try:
            compile(code, "<sandbox_meta>", "exec")
        except SyntaxError:
            return False
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os", "subprocess", "shutil", "socket", "ctypes"):
                            return False
                if isinstance(node, ast.ImportFrom):
                    if node.module in ("os", "subprocess", "shutil", "socket", "ctypes"):
                        return False
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ("system", "popen", "call", "Popen"):
                            return False
            return True
        except SyntaxError:
            return False

    async def _run_tests_in_sandbox(self, sandbox_path: Path) -> tuple[int, int]:
        """Run pytest in the sandbox. Returns (passed, failed)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                cwd=str(sandbox_path),
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + result.stderr
            # Parse pytest output
            import re
            passed_match = re.search(r"(\d+)\s+passed", output)
            failed_match = re.search(r"(\d+)\s+failed", output)
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            return passed, failed
        except subprocess.TimeoutExpired:
            return 0, 1
        except Exception as e:
            logger.warning("SandboxMeta: test run failed: %s", e)
            return 0, 1

    def _save_proposal(self, exp: SandboxExperiment):
        """Save approved proposal for human review."""
        proposals_dir = SANDBOX_DIR / "proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        proposal = {
            "experiment_id": exp.experiment_id,
            "target_file": exp.target_file,
            "diff": exp.diff,
            "proposed_changes": exp.proposed_changes[:5000],
            "tests_passed": exp.tests_passed,
            "tests_failed": exp.tests_failed,
            "safety_check": exp.safety_check,
            "timestamp": exp.timestamp,
            "status": "pending_review",
        }
        path = proposals_dir / f"{exp.experiment_id}.json"
        path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_experiment_log(self, exp: SandboxExperiment):
        log_path = SANDBOX_DIR / "experiments.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": exp.experiment_id,
                "target": exp.target_file,
                "applied": exp.applied,
                "tests_passed": exp.tests_passed,
                "tests_failed": exp.tests_failed,
                "safety": exp.safety_check,
                "error": exp.error,
                "timestamp": exp.timestamp,
            }, ensure_ascii=False) + "\n")

    def _cleanup(self, sandbox_path: Path):
        """Remove sandbox after experiment."""
        try:
            if sandbox_path.exists():
                shutil.rmtree(str(sandbox_path), ignore_errors=True)
        except Exception:
            pass

    def status(self) -> dict:
        proposals_dir = SANDBOX_DIR / "proposals"
        pending = list(proposals_dir.glob("*.json")) if proposals_dir.exists() else []
        return {
            "total_experiments": len(self._history),
            "successful": sum(1 for e in self._history if e.applied),
            "failed": sum(1 for e in self._history if not e.applied),
            "pending_proposals": len(pending),
            "recent": [
                {
                    "id": e.experiment_id, "target": e.target_file,
                    "applied": e.applied, "tests": f"{e.tests_passed}/{e.tests_passed + e.tests_failed}",
                }
                for e in self._history[-5:]
            ],
        }
