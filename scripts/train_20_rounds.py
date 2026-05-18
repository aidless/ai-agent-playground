"""20-round Self-Play Training — prove autonomous skill improvement.

Runs 20 rounds of self-play training across 7 domains. Tracks skill
curves per domain and generates a trend report proving whether the
agent genuinely improves through autonomous practice.

Usage: uv run python scripts/train_20_rounds.py
"""

import asyncio, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import dotenv
from openai import AsyncOpenAI

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
dotenv.load_dotenv(PROJECT / ".env")


async def main():
    deepseek = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    from agent.async_core import AsyncAgent
    from agent.self_play import SelfPlayEngine, DOMAINS
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("echo", "Echo", {"properties": {"text": {"type": "str"}}, "required": ["text"]}, lambda text: text)
    agent = AsyncAgent(client=deepseek, model="deepseek-chat", registry=registry,
                       enable_super_agent=True, enable_reflection=True, enable_learning=True)
    sp = SelfPlayEngine(agent, deepseek)

    rounds = 20
    snapshots = []
    print(f"Self-Play Training: {rounds} rounds across 7 domains")
    print(f"Start: {datetime.now(timezone.utc).isoformat()}")
    print()

    for i in range(0, rounds, 5):
        batch = min(5, rounds - i)
        print(f"Batch {i//5 + 1}: rounds {i+1}-{i+batch}...")
        results = await sp.train(rounds=batch, consolidate=(i >= 15))
        snapshots.append({
            "round": i + batch,
            "competence": sp.competence.status(),
            "insights": sp._strategy_insights[-3:] if sp._strategy_insights else [],
        })

        # Print progress
        status = sp.competence.status()
        for domain, info in sorted(status.items(), key=lambda x: x[1].get("skill", 0), reverse=True):
            if info["attempts"] > 0:
                bar = "█" * int(info["skill"])
                print(f"  {domain:25s} skill={info['skill']:4.1f} {bar}")

    # Final trend analysis
    print(f"\n{'='*70}")
    print("TREND ANALYSIS")
    print(f"{'='*70}")

    final_status = sp.competence.status()
    improvements = []
    for domain, info in final_status.items():
        if info["attempts"] >= 3:
            # Check if we have early data
            if len(snapshots) >= 2:
                early_skill = snapshots[0]["competence"].get(domain, {}).get("skill", 0)
                late_skill = info["skill"]
                delta = late_skill - early_skill
                trend = "↑ IMPROVING" if delta > 0.5 else "→ STABLE" if abs(delta) <= 0.5 else "↓ DECLINING"
                improvements.append((domain, delta, trend))
                print(f"  {domain:25s} {early_skill:.1f} → {late_skill:.1f}  delta={delta:+.1f}  {trend}")

    print(f"\nTotal rounds: {rounds}")
    print(f"Domains improved: {sum(1 for _, d, _ in improvements if d > 0.5)}")
    print(f"Domains stable: {sum(1 for _, d, _ in improvements if abs(d) <= 0.5)}")
    print(f"Domains declined: {sum(1 for _, d, _ in improvements if d < -0.5)}")

    overall_delta = sum(d for _, d, _ in improvements) / max(1, len(improvements))
    print(f"\nOverall skill delta: {overall_delta:+.1f}")
    print(f"Verdict: {'✅ Proves improvement' if overall_delta > 0 else '⚠ No clear improvement'}")

    # Save comprehensive report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
        "snapshots": snapshots,
        "final_competence": final_status,
        "improvements": [{"domain": d, "delta": delta, "trend": t} for d, delta, t in improvements],
        "overall_delta": round(overall_delta, 1),
        "total_lessons_learned": len(agent.memory.get_recent_lessons(100)),
    }
    report_path = PROJECT / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
