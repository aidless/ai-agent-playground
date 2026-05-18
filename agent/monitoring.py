"""Prometheus 监控指标"""

import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware


# ── 指标定义 ──

HTTP_REQUESTS_TOTAL = Counter(
    "agent_http_requests_total",
    "总请求数",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "agent_http_request_duration_seconds",
    "请求延迟（秒）",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

LLM_CALLS_TOTAL = Counter(
    "agent_llm_calls_total",
    "LLM 调用次数",
    ["model", "status"],
)

TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "工具调用次数",
    ["tool"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """记录请求数和延迟到 Prometheus 指标"""

    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response
