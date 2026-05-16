"""Eval Harness CLI — test your AI agents.

Usage:
  uv run python -m eval_harness.main              # Run all tests
  uv run python -m eval_harness.main hello        # Test only hello agent
  uv run python -m eval_harness.main --quick      # Fast checks only (no LLM judge)
  uv run python -m eval_harness.main --report md  # Save Markdown report
"""

import sys
from .runner import run_evaluation
from .reporter import print_report, save_markdown


def main():
    # Parse args
    agent_filter = None
    scorers = None  # None = use defaults (contains + llm_judge)
    save_md = False

    for arg in sys.argv[1:]:
        if arg == "--quick":
            scorers = ["contains"]  # Skip expensive LLM judge
        elif arg == "--report":
            save_md = True
        elif not arg.startswith("--"):
            agent_filter = arg

    print("=" * 70)
    print("  AI Agent Evaluation Harness")
    mode = "QUICK (keyword checks only)" if scorers == ["contains"] else "FULL (keyword + LLM judge)"
    print(f"  Mode: {mode}")
    if agent_filter:
        print(f"  Agent: {agent_filter}")
    print("=" * 70)

    # Run evaluation
    reports = run_evaluation(
        agent_filter=agent_filter,
        scorers=scorers,
        pass_threshold=0.6,
    )

    # Print results
    print_report(reports)

    # Save markdown report
    if save_md:
        path = save_markdown(reports)
        print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
