"""
Prompt Engineering Demo — registry, A/B test, few-shot selection.

Demonstrates:
  1. Prompt Registry: versioned prompt storage with diff
  2. Prompt A/B Test: quantitative comparison of two prompt versions
  3. Few-shot Selection: dynamic example selection via embedding similarity

Usage:
  uv run python -m demo.prompt_demo
"""

from ai_agent_playground.prompt_registry import (
    PromptRegistry,
    create_default_registry,
)
from ai_agent_playground.prompt_eval import PromptABTest
from ai_agent_playground.fewshot import (
    FewShotPool,
    FewShotSelector,
    create_calculator_pool,
    create_general_pool,
)


def demo_registry():
    """Demo 1: Prompt versioning and diff."""
    print("=" * 60)
    print("  Demo 1: Prompt Registry — Versioning & Diff")
    print("=" * 60)
    print()

    registry = create_default_registry()

    # List all prompts
    print("Registered prompts:")
    for name in registry.list_all():
        versions = registry.list_versions(name)
        print(f"  {name}: {', '.join(versions)}")
    print()

    # Show diff between v1 and v2 of mcp_agent prompt
    print("Diff: mcp_agent v1 → v2")
    print("-" * 40)
    diff = registry.diff("mcp_agent", "v1", "v2")
    print(diff or "(identical)")
    print()

    # Show prompt variable extraction
    template = registry.get_latest("mcp_agent")
    if template:
        vars_ = template._extract_variables()
        if vars_:
            print(f"Variables in mcp_agent: {vars_}")
        else:
            print("mcp_agent prompt has no template variables")
    print()


def demo_ab_test():
    """Demo 2: A/B test two prompt versions."""
    print("=" * 60)
    print("  Demo 2: Prompt A/B Test — v1 vs v2")
    print("=" * 60)
    print()

    registry = create_default_registry()

    # Test questions for the MCP agent prompt
    test_questions = [
        "What is 15 * 15 + 12? Calculate it.",
        "Explain what an AI agent is in 2 sentences.",
        "Should I use a calculator for 'what is the capital of France'?",
        "How do I read a file named config.json?",
    ]

    # Create A/B test without running LLM (too slow for demo)
    # Instead, show the test structure
    print(f"Test setup:")
    print(f"  Prompt: mcp_agent")
    print(f"  Versions: v1 vs v2")
    print(f"  Questions: {len(test_questions)}")
    print()

    print("v1 prompt (first 100 chars):")
    p1 = registry.get("mcp_agent", "v1")
    print(f"  \"{p1.content[:100]}...\"")
    print()

    print("v2 prompt (first 100 chars):")
    p2 = registry.get("mcp_agent", "v2")
    print(f"  \"{p2.content[:100]}...\"")
    print()

    print("To run the full A/B test (requires API calls):")
    print("  from ai_agent_playground.prompt_eval import PromptABTest")
    print("  test = PromptABTest(registry, 'mcp_agent', 'v1', 'v2')")
    print("  report = test.run(test_questions)")
    print("  test.print_report(report)")
    print()


def demo_fewshot():
    """Demo 3: Dynamic few-shot example selection."""
    print("=" * 60)
    print("  Demo 3: Few-Shot Selection — Dynamic Example Picking")
    print("=" * 60)
    print()

    # Calculator pool
    calc_pool = create_calculator_pool()
    print(f"Calculator pool: {len(calc_pool)} examples")

    selector = FewShotSelector(calc_pool)

    # Test queries
    queries = [
        "What is 50 * 3?",
        "Calculate the square root of 256",
        "Tell me about the weather",
    ]

    for q in queries:
        print(f"\n  Query: {q}")
        examples = selector.select(q, num_examples=2, strategy="semantic")
        for i, ex in enumerate(examples):
            print(f"    Example {i + 1}: {ex.input} → {ex.output[:60]}...")

    print()

    # Show a built few-shot prompt
    print("Full few-shot prompt for 'What is 50 * 3?':")
    print("-" * 40)
    prompt = selector.build_prompt(
        "What is 50 * 3?",
        num_examples=2,
        prefix="You are a math assistant. Use the calculator tool when needed.",
    )
    print(prompt)

    print()

    # General pool — shows tool selection diversity
    general_pool = create_general_pool()
    gen_selector = FewShotSelector(general_pool)

    print(f"General pool: {len(general_pool)} examples")

    test_queries = [
        "Calculate 100 / 3",
        "Read the README.md file",
        "Search for AI news",
    ]

    for q in test_queries:
        examples = gen_selector.select(q, num_examples=2, strategy="semantic")
        best = examples[0] if examples else None
        tool = best.output[:50] if best else "?"
        print(f"  '{q}' → closest match: {best.input if best else '?'} → {tool}...")


def main():
    demo_registry()
    demo_ab_test()
    demo_fewshot()

    print()
    print("=" * 60)
    print("  Done. Prompt Engineering demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
