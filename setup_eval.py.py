#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键生成 Agent 评测模块（兼容 Windows 路径与 UTF-8）"""
import os
from pathlib import Path

def write_file(filepath: Path, content: str):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已生成: {filepath.name}")

def main():
    target_dir = Path(r"C:\Users\Administrator\Desktop\ai-agent-playground")
    if not target_dir.exists():
        print(f"❌ 目标目录不存在: {target_dir}\n请确认路径或手动创建后重试。")
        return

    print(f"📂 目标目录: {target_dir}\n🚀 开始生成评测模块...")

    # 1. eval_runner.py
    write_file(target_dir / "eval_runner.py", r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, uuid, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class Trace:
    id: str
    question: str
    ground_truth: str
    agent_output: str
    latency_ms: float
    success: bool
    error: str
    score: float = 0.0
    judge_reason: str = ""

class EvalRunner:
    def __init__(self, agent, max_workers: int = 4, timeout: int = 30):
        self.agent = agent
        self.max_workers = max_workers
        self.timeout = timeout

    def run_batch(self, dataset_path: str, output_path: str = "traces.jsonl") -> List[Trace]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]

        traces = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_single, c): c for c in cases}
            for fut in as_completed(futures):
                case = futures[fut]
                try:
                    traces.append(fut.result(timeout=self.timeout))
                except Exception as e:
                    traces.append(Trace(
                        str(uuid.uuid4())[:8], case["question"], case.get("ground_truth", ""),
                        "", self.timeout * 1000, False, f"Timeout/ExecError: {e}"
                    ))

        with open(output_path, "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        return traces

    def _run_single(self, case: Dict[str, Any]) -> Trace:
        tid = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        try:
            model_in = self.agent.preprocess(case["question"])
            model_out = self.agent._forward(model_in)
            answer = self.agent.postprocess(model_out)
            latency = (time.perf_counter() - start) * 1000
            return Trace(tid, case["question"], case.get("ground_truth", ""), answer, latency, True, "")
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return Trace(tid, case["question"], case.get("ground_truth", ""), "", latency, False, traceback.format_exc())
''')

    # 2. scorer.py
    write_file(target_dir / "scorer.py", r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from typing import List

def rule_score(agent_output: str, keywords: List[str]) -> float:
    text = (agent_output or "").lower()
    return 1.0 if any(kw.lower() in text for kw in keywords) else 0.0

LLM_JUDGE_PROMPT = """你是一个严格的Agent评测裁判。请根据标准答案对模型输出打分（0.0~1.0）。
【问题】{question}
【标准答案】{ground_truth}
【模型输出】{agent_output}
请仅返回JSON：{{"score": 0.8, "reason": "简要说明"}}"""

def llm_judge_score(trace: dict, llm_call_fn) -> dict:
    prompt = LLM_JUDGE_PROMPT.format(**trace)
    res = llm_call_fn(prompt)
    try:
        return json.loads(re.search(r"\{.*\}", res, re.S).group())
    except Exception:
        return {"score": 0.0, "reason": "JSON解析失败"}
''')

    # 3. benchmark_dataset.jsonl
    write_file(target_dir / "benchmark_dataset.jsonl", '''{"question": "如何查询本周出港的货船列表？", "ground_truth": "调用 list_ships API，过滤 status='departed' 且 date 在本周", "keywords": ["list_ships", "departed"]}
{"question": "RAG 检索不到文档时 Agent 应如何降级？", "ground_truth": "返回预设兜底话术或触发人工接管，不应编造答案", "keywords": ["兜底", "人工接管"]}
''')

    # 4. run_eval.py
    write_file(target_dir / "run_eval.py", r'''#!/usr/bin/env python3
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
''')

    print("\n🎉 全部文件已就绪！")
    print("👉 下一步: 在终端执行以下命令")
    print("   chcp 65001")
    print("   cd /d \"C:\\Users\\Administrator\\Desktop\\ai-agent-playground\"")
    print("   python run_eval.py")

if __name__ == "__main__":
    main()