"""演示脚本：直接用 DeepSeek API 跑一次多Agent协作

会打印实时的执行报告，展示：
1. Crew 创建
2. 任务拆解
3. 每个 Agent 独立执行结果
4. 聚合 + 投票
5. 最终报告
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from agent.orchestrator import AgentOrchestrator, Crew
from agent.governance import GovernancePanel


async def main():
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    gov = GovernancePanel()
    crew = Crew(client, model="deepseek-chat")
    crew.add("developer-1", "developer")
    crew.add("reviewer-1", "reviewer")
    crew.add("tester-1", "tester")

    orch = AgentOrchestrator(client, model="deepseek-chat")

    # 换个实用任务
    task = "帮我在 agent/ 目录下创建一个新的工具模块 agent/tools/time_tool.py，功能是获取当前时间、计算两个日期之差。要求：1) 遵循现有 TOOLS 列表模式 2) 注册到 registry 3) 写单元测试"

    print("=" * 60)
    print("Multi-Agent Demo: 新工具模块开发")
    print(f"Crew: {[a.identity.role for a in crew.agents.values()]}")
    print(f"Task: {task[:80]}...")
    print("=" * 60)

    start = time.time()
    result = await orch.execute_with_crew(task, crew)
    elapsed = time.time() - start

    print(f"\nTotal: {elapsed:.1f}s | Subtasks: {result.subtask_count} | All OK: {all(r.success for r in result.agent_results)}")
    print(f"Agent Results:")
    for r in result.agent_results:
        tag = "OK" if r.success else "FAIL"
        print(f"  [{r.agent_name}] {r.latency_ms:.0f}ms {tag} | {r.content[:120]}")

    print(f"\n=== Final Output ===")
    print(result.final[:1000])

    # 治理
    print(f"\nGovernance: bus={crew.bus.message_count}msgs, audit={gov.audit.stats().get('total',0)}records")


if __name__ == "__main__":
    asyncio.run(main())
