"""
Production AI Systems Demo — observability, hallucination detection, eval gate.

Brings together all five enterprise-level capabilities:
  1. LLM Tracing + Metrics (observability.py)
  2. Hallucination Detection (hallucination_detector.py)
  3. Eval Gate (scripts/eval_gate.py)
  4. Cross-Encoder Re-ranking (reranker.py)
  5. Local LLM config (local_llm.py)

Usage:
  uv run python -m demo.production_demo
"""

import sys
from pathlib import Path


def demo_observability():
    """Demo 1: Trace an agent call with full observability."""
    print("=" * 60)
    print("  Demo 1: LLM Observability — Trace + Metrics")
    print("=" * 60)
    print()

    from ai_agent_playground.observability import get_tracer

    tracer = get_tracer(log_dir="logs/traces")

    # Simulate a traced agent interaction
    print("Simulating agent trace...")
    import time
    import random

    for i in range(3):
        with tracer.trace(f"agent_query_{i}", user_id="demo") as trace:
            with trace.span("llm_call", model="deepseek-v4-pro", tokens=random.randint(50, 200)):
                trace.spans[-1].attributes["output_tokens"] = random.randint(100, 400)
                time.sleep(random.uniform(0.1, 0.3))

            if random.random() > 0.3:
                with trace.span("tool_call", tool=random.choice(["calculator", "read_file"])):
                    time.sleep(random.uniform(0.05, 0.15))
                    if random.random() < 0.1:
                        raise RuntimeError("Tool timeout")
                    trace.spans[-1].end_time = time.time()

    # Print dashboard
    tracer.print_dashboard()

    # Export Prometheus metrics
    Path("logs/metrics").mkdir(parents=True, exist_ok=True)
    tracer.export_prometheus("logs/metrics/agent.prom")
    print("Prometheus metrics exported to logs/metrics/agent.prom")
    print()


def demo_hallucination():
    """Demo 2: Hallucination detection on sample answers."""
    print("=" * 60)
    print("  Demo 2: Hallucination Detection")
    print("=" * 60)
    print()

    from ai_agent_playground.hallucination_detector import HallucinationDetector

    detector = HallucinationDetector()

    # Test cases: good answer vs hallucinated answer
    test_cases = [
        {
            "name": "Good answer (fully supported)",
            "context": "Python 3.13 introduced a new interactive interpreter with multi-line editing and color support. It also added experimental JIT compilation via the copy-and-patch technique.",
            "answer": "Python 3.13 added multi-line editing, color support in the interactive interpreter, and experimental JIT compilation.",
        },
        {
            "name": "Bad answer (hallucinated)",
            "context": "Python 3.13 introduced a new interactive interpreter with multi-line editing and color support.",
            "answer": "Python 3.13 introduced GPU-accelerated computing and a new async runtime called Trio.",
        },
    ]

    for tc in test_cases:
        print(f"  Test: {tc['name']}")
        print(f"  Context: {tc['context'][:100]}...")
        print(f"  Answer:  {tc['answer'][:100]}...")
        print()

        # Fast check: only factual consistency (no API calls for demo)
        # For citation + self-consistency, uncomment the full check
        try:
            fact_score, claims = detector.check_factual_consistency(
                tc["answer"], tc["context"]
            )
            flagged = [c for c in claims if not c.supported]
            print(f"  Factual consistency: {fact_score:.2f}")
            if flagged:
                print(f"  Flagged claims ({len(flagged)}/{len(claims)}):")
                for c in flagged:
                    print(f"    ❌ {c.claim[:80]}...")

            if fact_score >= 0.8:
                print(f"  Verdict: SAFE — answer is consistent with context")
            elif fact_score >= 0.5:
                print(f"  Verdict: REVIEW — some claims may be unsupported")
            else:
                print(f"  Verdict: DANGER — answer likely contains hallucinations")
        except Exception as e:
            print(f"  Note: LLM-based check requires API ({e})")
        print()


def demo_eval_gate():
    """Demo 3: Eval gate — baseline + regression detection."""
    print("=" * 60)
    print("  Demo 3: Eval Gate — Baseline & Regression Detection")
    print("=" * 60)
    print()

    import json

    REPO_ROOT = Path(__file__).parent.parent
    baseline_path = REPO_ROOT / "reports" / "baseline.json"

    baseline = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    if baseline:
        print(f"Existing baseline ({len(baseline)} agents):")
        for agent, scores in baseline.items():
            print(f"  {agent}: {scores['avg_score']:.3f}")
    else:
        print("No baseline established yet.")
        print("Run the following to create one:")
        print("  uv run python scripts/eval_gate.py")

    print()
    print("Gate workflow:")
    print("  1. Commit code → eval gate runs automatically")
    print("  2. Scores compared against baseline")
    print("  3. If regression > 5% → merge blocked")
    print("  4. If scores improve → baseline auto-updated")
    print()
    print("Integration options:")
    print("  - Pre-commit hook:  cp scripts/eval_gate.py .git/hooks/pre-commit")
    print("  - GitHub Actions:   add eval_gate step to CI workflow")
    print("  - Manual:           uv run python scripts/eval_gate.py --ci")
    print()


def demo_reranker_availability():
    """Demo 4: Check reranker availability."""
    print("=" * 60)
    print("  Demo 4: Cross-Encoder Re-ranker")
    print("=" * 60)
    print()

    from rag_qa_system.reranker import create_reranker

    reranker = create_reranker()
    if reranker:
        print(f"Re-ranker available: {reranker.model_name}")
        print("Run retrieval eval with --compare to see hybrid+rerank scores:")
        print("  uv run python -m rag_qa_system.eval_retrieval --compare")
    else:
        print("Re-ranker not available (sentence-transformers not installed)")
        print("Install with: uv add sentence-transformers")
        print("Models: BAAI/bge-reranker-v2-m3 (multilingual, ~2GB)")
        print("        BAAI/bge-reranker-base      (English, ~1GB)")
    print()


def demo_local_llm():
    """Demo 5: Local LLM configuration guide."""
    print("=" * 60)
    print("  Demo 5: Local LLM Deployment")
    print("=" * 60)
    print()

    from ai_agent_playground.local_llm import (
        LocalLLMConfig,
        BackendType,
        Quantization,
        quantization_guide,
    )

    # Show recommended configs for different scenarios
    configs = {
        "GPU Server (>=16GB VRAM)": LocalLLMConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            backend=BackendType.VLLM,
            quantization=Quantization.AWQ,
            vllm_tensor_parallel=1,
        ),
        "GPU Workstation (8GB VRAM)": LocalLLMConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            backend=BackendType.VLLM,
            quantization=Quantization.GPTQ,
        ),
        "CPU Server (64GB RAM)": LocalLLMConfig(
            model="Qwen/Qwen2.5-7B-Instruct-GGUF",
            backend=BackendType.LLAMA_CPP,
            quantization=Quantization.GGUF_Q5,
            llama_n_threads=16,
        ),
        "Developer Laptop (16GB RAM)": LocalLLMConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            backend=BackendType.OLLAMA,
            quantization=Quantization.GGUF_Q4,
        ),
    }

    for scenario, cfg in configs.items():
        print(f"  {scenario}:")
        print(f"    Backend: {cfg.backend.value}")
        print(f"    Quant:   {cfg.quantization.value}")
        print(f"    Model:   {cfg.model}")
        print()

    print(quantization_guide())


def main():
    demos = [
        ("1. Observability", demo_observability),
        ("2. Hallucination Detection", demo_hallucination),
        ("3. Eval Gate", demo_eval_gate),
        ("4. Re-ranker", demo_reranker_availability),
        ("5. Local LLM", demo_local_llm),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            print(f"\n  [{name} skipped: {e}]\n")

    print("=" * 60)
    print("  Production AI Systems Demo Complete")
    print("=" * 60)
    print()
    print("Summary of enterprise capabilities now available:")
    print("  1. Tracing + Metrics (OpenTelemetry-compatible)")
    print("  2. Hallucination Detection (3-strategy layered defense)")
    print("  3. Eval Gate (CI automation, regression blocking)")
    print("  4. Cross-Encoder Re-ranking (BGE-Reranker)")
    print("  5. Local LLM Deployment (vLLM, llama.cpp, Ollama)")
    print("  6. Hybrid Search (BM25 + Vector RRF)")
    print("  7. Streaming + Concurrency")
    print("  8. Prompt Versioning + A/B Testing")


if __name__ == "__main__":
    main()
