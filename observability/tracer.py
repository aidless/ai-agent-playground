
import logging, json

logger = logging.getLogger("agent.trace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

def log_trace(trace_id: str, step: int, event: str, data: dict):
    logger.info(json.dumps({"trace_id": trace_id, "step": step, "event": event, **data}, ensure_ascii=False))
