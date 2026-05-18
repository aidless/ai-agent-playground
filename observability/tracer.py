"""链路追踪 — 支持结构化持久化和日志输出双通道"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("agent.trace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

# 结构化追踪目录
TRACES_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "memory", "traces",
)


def _ensure_traces_dir():
    os.makedirs(TRACES_DIR, exist_ok=True)


def log_trace(trace_id: str, step: int, event: str, data: dict):
    """记录追踪事件到日志 + 结构化文件"""
    entry = {
        "trace_id": trace_id,
        "step": step,
        "event": event,
        **data,
    }
    # 实时日志
    logger.info(json.dumps(entry, ensure_ascii=False))

    # 追加到结构化追踪文件
    _ensure_traces_dir()
    trace_path = os.path.join(TRACES_DIR, f"{trace_id}.jsonl")
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(), **entry}, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 追踪文件写入失败不应影响主流程
