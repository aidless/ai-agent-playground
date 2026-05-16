"""
Harness Engineering Demo — the 4 missing modules.

Demonstrates:
  1. State Manager — task checklist, journal, checkpoint, resume
  2. Human-in-the-Loop — risk-graded approval gates
  3. Evaluator Agent — online Generate/Evaluate separation
  4. Deterministic Constraints — programmatic guardrails

Usage:
  uv run python -m demo.harness_demo
"""

import tempfile
import sys
from pathlib import Path


def demo_state_manager():
    """Demo 1: State Manager — externalized memory for long-running agents."""
    print("=" * 60)
    print("  Demo 1: State Manager")
    print("=" * 60)
    print()

    from ai_agent_playground.state_manager import StateManager

    with tempfile.TemporaryDirectory(prefix="agent_state_") as tmp:
        sm = StateManager(work_dir=tmp)

        # Start a session
        sm.start_session("Build a user authentication API with JWT tokens")

        # Add tasks like Anthropic's task checklist
        sm.add_tasks([
            ("define_models", "Define User and Role database models"),
            ("create_auth", "Implement JWT token generation and validation"),
            ("add_routes", "Create POST /login and POST /register endpoints"),
            ("add_tests", "Write unit tests for auth flow"),
            ("add_docs", "Document API endpoints in OpenAPI format"),
        ])

        # Simulate agent working
        sm.start_task("define_models")
        sm.complete_task("define_models", "Created User(id, email, password_hash) and Role(id, name)")

        sm.start_task("create_auth")
        sm.complete_task("create_auth", "Implemented JWT with HS256, 1h expiry, refresh tokens")

        sm.start_task("add_routes")
        sm.fail_task("add_routes", "Test server port conflict — need to fix port binding")

        sm.skip_task("add_docs", "Deprioritized — will do after core works")

        print()
        print("Task status:")
        for item in sm.manifest.items:
            icon = {"completed": "✅", "in_progress": "🔄", "failed": "❌",
                    "pending": "⬜", "skipped": "⏭️"}[item.status]
            print(f"  {icon} [{item.id}] {item.description}")
        print(f"\n  Progress: {sm.manifest.completed_count}/{sm.manifest.total_count} "
              f"({sm.manifest.progress_pct:.0%})")

        # Show context rebuild
        print(f"\n  Context rebuild (what agent sees on restart):")
        context = sm.rebuild_context()
        for line in context.split("\n")[:15]:
            print(f"    {line}")

        print()
        print(f"  Manifest: {sm.manifest_path}")
        print(f"  Journal:  {sm.journal_path}")
        print()


def demo_human_loop():
    """Demo 2: Human-in-the-Loop — approval gates."""
    print("=" * 60)
    print("  Demo 2: Human-in-the-Loop — Approval Gates")
    print("=" * 60)
    print()

    from ai_agent_playground.human_loop import (
        ApprovalGate, Policy, assess_risk, DANGEROUS_COMMAND_PATTERNS,
    )

    gate = ApprovalGate()

    # Configure policies
    gate.set_policies({
        "read_file": Policy.AUTO_APPROVE,
        "calculator": Policy.AUTO_APPROVE,
        "write_file": Policy.ALWAYS_ASK,
        "run_command": Policy.ALWAYS_ASK,
        "delete_file": Policy.NEVER,
    })

    # Simulate tool calls with auto-approved input
    test_calls = [
        ("read_file", {"path": "config.json"}),
        ("calculator", {"expression": "15*15"}),
        ("write_file", {"path": "notes.txt", "content": "hello"}),
        ("run_command", {"command": "ls -la"}),
        ("run_command", {"command": "rm -rf /var/data"}),
        ("delete_file", {"path": "production.db"}),
    ]

    approved_count = 0
    denied_count = 0

    for tool_name, args in test_calls:
        risk = assess_risk(tool_name, args)
        # Use a mock input that auto-denies (for automated demo)
        decision = gate.approve(
            tool_name, args,
            input_fn=lambda _: "n",  # Auto-deny for demo
        )
        gate.log(tool_name, args, "simulated_result", decision.approved)

        status = "✅" if decision.approved else "❌"
        if decision.approved:
            approved_count += 1
        else:
            denied_count += 1

        print(f"  {status} {tool_name} (risk={risk}): {decision.reason}")

    gate.print_report()

    print()
    print("Dangerous command patterns detected:")
    print(f"  {DANGEROUS_COMMAND_PATTERNS[:8]}...")
    print()


def demo_evaluator():
    """Demo 3: Online Evaluator — Generate/Evaluate separation."""
    print("=" * 60)
    print("  Demo 3: Online Evaluator (Generate/Evaluate Separation)")
    print("=" * 60)
    print()

    from ai_agent_playground.evaluator_agent import (
        EvaluatorAgent, evaluate_agent_output,
        check_output_not_empty, check_no_error_keywords,
    )

    # Deterministic checks (no API calls)
    print("Deterministic checks (no LLM):")

    good_output = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

    bad_output = ""

    for label, output in [("Good output", good_output), ("Empty output", bad_output)]:
        print(f"  {label}:")
        not_empty = check_output_not_empty(output)
        no_err = check_no_error_keywords(output)
        print(f"    not_empty: {'✅' if not_empty.passed else '❌'} {not_empty.feedback}")
        print(f"    no_error_keywords: {'✅' if no_err.passed else '❌'} {no_err.feedback}")

    print()
    print("Online evaluator (requires API):")
    print("  from ai_agent_playground.evaluator_agent import EvaluatorAgent")
    print("  evaluator = EvaluatorAgent()")
    print("  verdict = evaluator.check(")
    print("      task='Write a Fibonacci function',")
    print("      output='def fib(n): return fib(n-1)+fib(n-2)  # no base case!',")
    print("  )")
    print()


def demo_constraints():
    """Demo 4: Deterministic Constraints."""
    print("=" * 60)
    print("  Demo 4: Deterministic Constraints — Programmatic Guardrails")
    print("=" * 60)
    print()

    from ai_agent_playground.constraints import (
        ConstraintRunner,
        NotEmptyConstraint,
        MinLengthConstraint,
        MustContainConstraint,
        MustNotContainConstraint,
        ValidJsonConstraint,
        JsonSchemaConstraint,
        PythonSyntaxConstraint,
        code_output_runner,
        json_output_runner,
    )

    # Code output runner
    print("Code Output Runner:")
    runner = code_output_runner()

    valid_code = "def hello():\n    return 'world'"
    invalid_code = "def hello(\n    return 'world'"  # syntax error

    for label, code in [("Valid code", valid_code), ("Syntax error", invalid_code)]:
        result = runner.check_all(code)
        status = "✅" if result.passed else "❌"
        violations = ", ".join(v.message for v in result.violations)
        print(f"  {status} {label}: {violations if violations else 'OK'}")

    print()

    # JSON output runner
    print("JSON Output Runner:")
    json_runner = json_output_runner(required_fields=["tool", "args"])

    valid_json = '{"tool": "calculator", "args": {"expression": "2+2"}}'
    bad_json = 'not json at all'
    missing_field = '{"tool": "calculator"}'

    for label, text in [("Valid JSON", valid_json), ("Not JSON", bad_json),
                         ("Missing field", missing_field)]:
        result = json_runner.check_all(text)
        status = "✅" if result.passed else "❌"
        violations = ", ".join(v.message for v in result.violations)
        print(f"  {status} {label}: {violations if violations else 'OK'}")

    print()


def main():
    demos = [
        ("1. State Manager", demo_state_manager),
        ("2. Human-in-the-Loop", demo_human_loop),
        ("3. Online Evaluator", demo_evaluator),
        ("4. Deterministic Constraints", demo_constraints),
    ]

    for name, fn in demos:
        fn()

    print("=" * 60)
    print("  Harness Engineering Demo Complete")
    print("=" * 60)
    print()
    print("Harness modules now available:")
    print("  1. State Manager    — task checklist + journal + checkpoints + resume")
    print("  2. Human-in-the-Loop — risk-graded approval gates + audit log")
    print("  3. Evaluator Agent  — online Generate/Evaluate separation, 4 checks")
    print("  4. Constraints      — 10 constraint types, factory pre-sets")
    print()
    print("Anthropic's 3-component architecture now representable:")
    print("  Planner → Generator → Evaluator → (loop back to Generator)")
    print("  All mediated by StateManager journal + ApprovalGate safety brakes")


if __name__ == "__main__":
    main()
