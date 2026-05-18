#!/usr/bin/env python3
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
