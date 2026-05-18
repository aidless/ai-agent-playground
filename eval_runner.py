#!/usr/bin/env python3
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
