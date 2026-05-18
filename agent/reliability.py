"""可靠性追踪模块 — CLEAR 模型：Reliability

追踪任务一致性：多次相同任务的成功率、波动率。
发现问题：单次 60%、重复 8 次降到 25% → 自动告警。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACKING_DIR = Path(__file__).resolve().parent.parent / "memory" / "reliability"


class ReliabilityTracker:
    """任务可靠性追踪器"""

    def __init__(self):
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []

    def record(self, task_type: str, task_input: str, success: bool, latency_ms: float, tool_calls: int = 0):
        """记录一次任务执行"""
        entry = {
            "ts": datetime.now().isoformat(),
            "task_type": task_type,
            "task_input_hash": str(hash(task_input)),
            "success": success,
            "latency_ms": latency_ms,
            "tool_calls": tool_calls,
        }
        self._history.append(entry)

        # 追加到文件
        path = TRACKING_DIR / "reliability.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_metrics(self, task_type: Optional[str] = None) -> dict:
        """计算可靠性指标"""
        records = self._load_records(task_type)

        if not records:
            return {"error": "no data", "samples": 0}

        total = len(records)
        successes = sum(1 for r in records if r.get("success"))
        failures = total - successes

        # 成功率
        success_rate = successes / total if total > 0 else 0

        # 连续失败检测（最近的连续失败次数）
        consecutive_fails = 0
        for r in reversed(records):
            if not r.get("success"):
                consecutive_fails += 1
            else:
                break

        # 一致性：相同输入的输出是否稳定
        consistency = self._calc_consistency(records)

        # 趋势：最近 3 次的成功率 vs 全部成功率
        recent = records[-3:] if len(records) >= 3 else records
        recent_rate = sum(1 for r in recent if r.get("success")) / len(recent) if recent else 0
        trend = "improving" if recent_rate > success_rate else "declining" if recent_rate < success_rate else "stable"

        return {
            "samples": total,
            "success_rate": round(success_rate, 3),
            "failures": failures,
            "consecutive_fails": consecutive_fails,
            "consistency_score": round(consistency, 3),
            "trend": trend,
            "recent_rate": round(recent_rate, 3),
            "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in records) / total, 0),
            "stability": "critical" if consecutive_fails >= 3 else "warning" if consecutive_fails >= 1 else "stable",
        }

    def detect_degradation(self) -> Optional[str]:
        """检测退化：重复 8 次后成功率是否显著下降"""
        records = self._load_records()
        if len(records) < 8:
            return None

        first_4 = records[:4]
        last_4 = records[-4:]

        first_rate = sum(1 for r in first_4 if r.get("success")) / 4
        last_rate = sum(1 for r in last_4 if r.get("success")) / 4

        degradation = first_rate - last_rate
        if degradation > 0.3:
            return f"可靠性退化警报: 成功率从 {first_rate:.0%} 降至 {last_rate:.0%} (落差 {degradation:.0%})"
        return None

    def _load_records(self, task_type: Optional[str] = None) -> list[dict]:
        path = TRACKING_DIR / "reliability.jsonl"
        if not path.exists():
            return []

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if task_type and r.get("task_type") != task_type:
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _calc_consistency(records: list[dict]) -> float:
        """计算一致性：相同 task_type 下成功率的标准差（越小越稳定）"""
        if len(records) < 2:
            return 1.0

        # 按 task_type 分组
        groups: dict[str, list[bool]] = {}
        for r in records:
            t = r.get("task_type", "unknown")
            groups.setdefault(t, []).append(r.get("success", False))

        # 计算每个组的成功率
        rates = [sum(v) / len(v) for v in groups.values() if len(v) >= 2]
        if not rates:
            return 0.0

        avg = sum(rates) / len(rates)
        variance = sum((r - avg) ** 2 for r in rates) / len(rates)
        return 1.0 - min(variance * 4, 1.0)  # 归一化到 [0, 1]
