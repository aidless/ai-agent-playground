#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from eval_runner import EvalRunner
from ai_agent_playground.base import BaseAgent

class EvalAgent(BaseAgent):
    def preprocess(self, inputs, **kwargs):
        return {"messages": [{"role": "user", "content": str(inputs)}]}

    def _forward(self, model_inputs, **kwargs):
        # 🔧 TODO: 替换为你的真实 LLM/工具调用逻辑
        return {"reply": f"Agent模拟回答: {model_inputs['messages'][0]['content']}"}

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs["reply"]

if __name__ == "__main__":
    agent = EvalAgent()
    runner = EvalRunner(agent, max_workers=2, timeout=30)
    print("🚀 开始批量评测...")
    traces = runner.run_batch("benchmark_dataset.jsonl", "traces.jsonl")

    print(f"\n✅ 评测完成: 共 {len(traces)} 条")
    for t in traces:
        status = "✅" if t.success else "❌"
        print(f"{status} [{t.id}] 耗时:{t.latency_ms:.0f}ms | 输出:{t.agent_output[:50]}...")
