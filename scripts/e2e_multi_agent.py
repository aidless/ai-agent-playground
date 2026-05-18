"""端到端多Agent协作测试 — 使用真实 DeepSeek API

任务流程:
    1. 创建 Crew (developer + reviewer + tester + master)
    2. Master 拆解任务
    3. Worker 并行执行（独立上下文）
    4. Master 聚合 + 投票
    5. 输出完整执行报告
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from openai import AsyncOpenAI
from agent.orchestrator import AgentOrchestrator, Crew
from agent.governance import GovernancePanel
from agent.memory import get_memory


async def main():
    # 1. 初始化
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    gov = GovernancePanel()
    mem = get_memory()

    print("=" * 60)
    print("Multi-Agent E2E Test — DeepSeek API")
    print("=" * 60)

    # 2. 创建 Crew
    crew = Crew(client, model="deepseek-chat")
    crew.add("developer-1", "developer")
    crew.add("reviewer-1", "reviewer")
    crew.add("tester-1", "tester")
    # Master 自动添加

    print(f"\nCrew: {[a.stats()['name'] + '(' + a.stats()['role'] + ')' for a in crew.agents.values()]}")
    print(f"Bus registered: {crew.bus.list_agents()}")

    # 3. 执行任务
    task = "设计一个简单的用户注册 API：POST /register，接收 username+password，返回 JWT token。请拆解为子任务并执行。"

    orch = AgentOrchestrator(client, model="deepseek-chat")

    print(f"\n>>> Task: {task}")
    print(f"\n--- Executing ---")
    start = time.time()

    result = await orch.execute_with_crew(task, crew)

    elapsed = time.time() - start

    # 4. 输出报告
    print(f"\n=== Results ({elapsed:.1f}s) ===")
    print(f"Subtasks: {result.subtask_count}")
    print(f"Consensus: {json.dumps(result.consensus, ensure_ascii=False)}")
    print()

    for i, r in enumerate(result.agent_results):
        status = "OK" if r.success else f"FAIL: {r.error}"
        print(f"--- Agent [{r.agent_name}] ({r.latency_ms:.0f}ms) {status} ---")
        # 安全截断（避免编码问题）
        safe_content = r.content[:300].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(safe_content)
        print()

    print("=== Final Synthesis ===")
    safe_final = result.final[:800].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    print(safe_final)
    print()

    # 5. 治理报告
    print("=== Governance ===")
    print(f"Messages on bus: {crew.bus.message_count}")
    print(f"Audit records: {gov.audit.stats().get('total', 0)}")
    for a in crew.agents.values():
        s = a.stats()
        print(f"  {s['name']}: {s['tasks_completed']} tasks, {s['avg_latency_ms']:.0f}ms avg")

    # 保存结果
    report_path = Path(__file__).resolve().parent.parent / "memory" / "e2e_multi_agent_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "elapsed_s": elapsed,
        "subtask_count": result.subtask_count,
        "consensus": result.consensus,
        "results": [
            {
                "agent": r.agent_name,
                "success": r.success,
                "content": r.content[:500],
                "latency_ms": r.latency_ms,
            }
            for r in result.agent_results
        ],
        "final": result.final[:1000],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
