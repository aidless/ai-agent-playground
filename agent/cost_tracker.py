"""成本追踪模块 — CLEAR 模型：Cost

跟踪每次 LLM 调用的 Token 用量、估算费用、预算熔断。
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACKING_DIR = Path(__file__).resolve().parent.parent / "memory" / "cost"

# 价格参考（实际以各平台官方定价为准，RMB/1K tokens，汇率约 7:1）
PRICING = {
    "deepseek-chat":       {"input": 0.00014, "output": 0.00028},   # DeepSeek V3
    "deepseek-reasoner":   {"input": 0.00055, "output": 0.00219},   # DeepSeek R1
    "claude-sonnet-4-6":   {"input": 0.021,   "output": 0.105},     # Claude Sonnet
    "claude-opus-4-7":     {"input": 0.105,   "output": 0.525},     # Claude Opus
    "qwen2.5:7b":          {"input": 0.0,     "output": 0.0},        # 本地免费
}


class BudgetCap:
    """预算熔断器"""

    def __init__(self, daily_limit_usd: float = 5.0, monthly_limit_usd: float = 50.0):
        self.daily_limit = daily_limit_usd
        self.monthly_limit = monthly_limit_usd

    def check(self, daily_total: float, monthly_total: float) -> tuple[bool, str]:
        """检查是否超预算，返回 (允许继续, 原因)"""
        if daily_total >= self.daily_limit:
            return False, f"日预算已超: ${daily_total:.4f} >= ${self.daily_limit:.2f}"
        if monthly_total >= self.monthly_limit:
            return False, f"月预算已超: ${monthly_total:.4f} >= ${self.monthly_limit:.2f}"
        return True, "ok"


class CostTracker:
    """Token 用量和费用追踪"""

    def __init__(self, model: str = "deepseek-chat", budget: Optional[BudgetCap] = None):
        self.model = model
        self.budget = budget or BudgetCap()
        self._current_request: dict = {"input_tokens": 0, "output_tokens": 0}
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    def record_input(self, tokens: int):
        self._current_request["input_tokens"] += tokens

    def record_output(self, tokens: int):
        self._current_request["output_tokens"] += tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """估算本次请求费用（USD）"""
        prices = PRICING.get(self.model, {"input": 0.001, "output": 0.002})
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000

    def get_daily_total(self) -> float:
        return self._read_today_file().get("total_cost_usd", 0.0)

    def get_monthly_total(self) -> float:
        today = datetime.now()
        total = 0.0
        tracking_dir = TRACKING_DIR
        for f in tracking_dir.glob(f"cost-{today.year}-{today.month:02d}-*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                total += data.get("total_cost_usd", 0.0)
            except Exception:
                continue
        return total

    def pre_check(self) -> bool:
        """调用前检查预算"""
        ok, reason = self.budget.check(self.get_daily_total(), self.get_monthly_total())
        if not ok:
            logger.warning("预算熔断: %s", reason)
            return False
        return True

    def commit(self, input_tokens: int = 0, output_tokens: int = 0):
        """提交本次请求用量并持久化"""
        inp = input_tokens or self._current_request["input_tokens"]
        out = output_tokens or self._current_request["output_tokens"]
        cost = self.estimate_cost(inp, out)

        today = datetime.now()
        filename = TRACKING_DIR / f"cost-{today.strftime('%Y-%m-%d')}.json"

        data = self._read_today_file()
        data["date"] = today.strftime("%Y-%m-%d")
        data["total_tokens"] = data.get("total_tokens", 0) + inp + out
        data["total_cost_usd"] = data.get("total_cost_usd", 0.0) + cost
        data["requests"] = data.get("requests", 0) + 1
        data["model"] = self.model

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._current_request = {"input_tokens": 0, "output_tokens": 0}
        logger.info("Cost: +$%.4f (daily total: $%.4f)", cost, data["total_cost_usd"])

    def summary(self) -> dict:
        daily = self._read_today_file()
        return {
            "model": self.model,
            "today_tokens": daily.get("total_tokens", 0),
            "today_cost_usd": daily.get("total_cost_usd", 0.0),
            "today_requests": daily.get("requests", 0),
            "monthly_cost_usd": self.get_monthly_total(),
            "budget_daily_limit": self.budget.daily_limit,
            "budget_monthly_limit": self.budget.monthly_limit,
            "price_per_1k_tokens": PRICING.get(self.model, {}),
        }

    def _read_today_file(self) -> dict:
        today = datetime.now()
        filename = TRACKING_DIR / f"cost-{today.strftime('%Y-%m-%d')}.json"
        if filename.exists():
            return json.loads(filename.read_text(encoding="utf-8"))
        return {}
