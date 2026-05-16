"""
Deterministic Constraints — programmatic guardrails that don't rely on LLMs.

The article's key insight: "don't rely on the model's self-discipline —
use deterministic linters and structural tests to enforce rules."

A LLM judge is itself a model — it can hallucinate, be too lenient, or miss issues.
Deterministic constraints are pure code: they always run the same way,
produce the same result, and catch errors that models might miss.

Constraint types:
  - Structural:     output must follow a specific format (JSON, Markdown, etc.)
  - Pattern:         output must match (or not match) regex patterns
  - Code quality:    run linters on generated code
  - Schema:          validate JSON output against a schema
  - Bounds:          output must be within length, token, or time limits

Usage:
    from ai_agent_playground.constraints import ConstraintRunner

    runner = ConstraintRunner()
    runner.add(JsonValidConstraint())
    runner.add(NoMarkdownCodeBlockConstraint())  # must NOT fence code
    runner.add(MaxTokensConstraint(2048))

    result = runner.check_all(output)
    if not result.passed:
        for violation in result.violations:
            print(f"Constraint violation: {violation}")
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ConstraintViolation:
    constraint: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class ConstraintResult:
    passed: bool
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")


class Constraint(ABC):
    """Abstract constraint. All constraints produce ConstraintResult."""

    @abstractmethod
    def check(self, output: str) -> ConstraintResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ============================================================
#  Structural constraints
# ============================================================


class NotEmptyConstraint(Constraint):
    """Output must not be empty or whitespace-only."""
    name = "not_empty"

    def check(self, output: str) -> ConstraintResult:
        if not output or not output.strip():
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(self.name, "Output is empty", "error")
            ])
        return ConstraintResult(passed=True)


class MinLengthConstraint(Constraint):
    """Output must be at least N characters long."""
    name = "min_length"

    def __init__(self, min_length: int = 20):
        self.min_length = min_length

    def check(self, output: str) -> ConstraintResult:
        if len(output) < self.min_length:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Output too short: {len(output)} < {self.min_length}",
                    "error",
                )
            ])
        return ConstraintResult(passed=True)


class MaxLengthConstraint(Constraint):
    """Output must not exceed N characters."""
    name = "max_length"

    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def check(self, output: str) -> ConstraintResult:
        if len(output) > self.max_length:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Output too long: {len(output)} > {self.max_length}",
                    "warning",
                )
            ])
        return ConstraintResult(passed=True)


# ============================================================
#  Pattern constraints
# ============================================================


class MustContainConstraint(Constraint):
    """Output must contain specific substrings or regex patterns."""
    name = "must_contain"

    def __init__(self, patterns: list[str], match_all: bool = False):
        self.patterns = patterns
        self.match_all = match_all

    def check(self, output: str) -> ConstraintResult:
        missing = []
        for pat in self.patterns:
            if not re.search(pat, output):
                missing.append(pat)
            elif not self.match_all:
                # Found one — if match_all=False, one match is enough
                return ConstraintResult(passed=True)

        if self.match_all and not missing:
            return ConstraintResult(passed=True)

        if missing:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Missing patterns: {missing}",
                    "error",
                )
            ])
        return ConstraintResult(passed=True)


class MustNotContainConstraint(Constraint):
    """Output must NOT contain specific substrings or regex patterns."""
    name = "must_not_contain"

    def __init__(self, forbidden: list[str]):
        self.forbidden = forbidden

    def check(self, output: str) -> ConstraintResult:
        found = []
        for pat in self.forbidden:
            if re.search(pat, output):
                found.append(pat)

        if found:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Forbidden patterns found: {found}",
                    "warning",
                )
            ])
        return ConstraintResult(passed=True)


class NoMarkdownCodeBlockConstraint(Constraint):
    """Output must NOT contain Markdown code fences.
    Useful when the output is supposed to be raw code, not wrapped.
    """
    name = "no_markdown_code_block"

    def check(self, output: str) -> ConstraintResult:
        if re.search(r'```', output):
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    "Output contains Markdown code fences — should be raw text",
                    "warning",
                )
            ])
        return ConstraintResult(passed=True)


# ============================================================
#  JSON / Schema constraints
# ============================================================


class ValidJsonConstraint(Constraint):
    """Output must be valid JSON."""
    name = "valid_json"

    def check(self, output: str) -> ConstraintResult:
        text = output.strip()
        # Try the whole output
        try:
            json.loads(text)
            return ConstraintResult(passed=True)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return ConstraintResult(passed=True)
            except json.JSONDecodeError:
                pass

        return ConstraintResult(passed=False, violations=[
            ConstraintViolation(self.name, "Output is not valid JSON", "error")
        ])


class JsonSchemaConstraint(Constraint):
    """Output must be valid JSON matching a given schema."""
    name = "json_schema"

    def __init__(self, required_fields: list[str]):
        self.required_fields = required_fields

    def check(self, output: str) -> ConstraintResult:
        # First, validate JSON
        valid_json = ValidJsonConstraint().check(output)
        if not valid_json.passed:
            return valid_json

        # Extract and validate schema
        text = output.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                return ConstraintResult(passed=False, violations=[
                    ConstraintViolation(self.name, "Cannot extract JSON", "error")
                ])

        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Missing required fields: {missing}",
                    "error",
                )
            ])
        return ConstraintResult(passed=True)


# ============================================================
#  Code quality constraints (require external tools)
# ============================================================


class PythonSyntaxConstraint(Constraint):
    """Output must be syntactically valid Python code."""
    name = "python_syntax"

    def check(self, output: str) -> ConstraintResult:
        # Strip markdown code fences if present
        code = output
        match = re.search(r'```(?:python)?\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            code = match.group(1)

        try:
            compile(code, "<agent_output>", "exec")
            return ConstraintResult(passed=True)
        except SyntaxError as e:
            return ConstraintResult(passed=False, violations=[
                ConstraintViolation(
                    self.name,
                    f"Invalid Python syntax: {e.msg} (line {e.lineno})",
                    "error",
                )
            ])


# ============================================================
#  Constraint runner — combines multiple constraints
# ============================================================


class ConstraintRunner:
    """Run multiple constraints against agent output.

    This is the gate that sits between "agent produced something" and
    "we show it to the user / commit it / deploy it."

    Usage:
        runner = ConstraintRunner()
        runner.add(NotEmptyConstraint())
        runner.add(PythonSyntaxConstraint())

        result = runner.check_all(code_output)
        if not result.passed:
            print(f"Blocked: {result.error_count} errors")
            return  # Don't proceed
    """

    def __init__(self):
        self.constraints: list[Constraint] = []

    def add(self, constraint: Constraint):
        self.constraints.append(constraint)

    def check_all(self, output: str) -> ConstraintResult:
        """Run all constraints. Short-circuits on first error by default."""
        all_violations = []
        for c in self.constraints:
            result = c.check(output)
            all_violations.extend(result.violations)
            if result.violations and any(
                v.severity == "error" for v in result.violations
            ):
                break  # Short-circuit on errors

        passed = not any(v.severity == "error" for v in all_violations)
        return ConstraintResult(passed=passed, violations=all_violations)


# ============================================================
#  Factory — pre-built constraint sets
# ============================================================


def code_output_runner() -> ConstraintRunner:
    """Constraint set for agent-generated code."""
    runner = ConstraintRunner()
    runner.add(NotEmptyConstraint())
    runner.add(MinLengthConstraint(10))
    runner.add(NoMarkdownCodeBlockConstraint())  # Code should be raw
    runner.add(PythonSyntaxConstraint())
    return runner


def json_output_runner(required_fields: list[str] | None = None) -> ConstraintRunner:
    """Constraint set for agent-generated JSON."""
    runner = ConstraintRunner()
    runner.add(NotEmptyConstraint())
    runner.add(ValidJsonConstraint())
    if required_fields:
        runner.add(JsonSchemaConstraint(required_fields))
    return runner


def safe_content_runner() -> ConstraintRunner:
    """Constraint set for safe content output."""
    runner = ConstraintRunner()
    runner.add(NotEmptyConstraint())
    runner.add(MustNotContainConstraint([
        r'<script', r'javascript:', r'onerror\s*=',
        r'DROP\s+TABLE', r'DELETE\s+FROM',
    ]))
    return runner
