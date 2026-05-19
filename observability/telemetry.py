"""OpenTelemetry Integration — distributed tracing for agent steps.

Integrates with existing tracer.py. Exports spans to OTLP collector
(Jaeger/Grafana/Prometheus) for full-stack observability.

Usage:
    from observability.telemetry import AgentTelemetry

    telemetry = AgentTelemetry("ai-agent")

    @telemetry.trace_step("PLANNING")
    async def planning(self, goal):
        ...

Attributes tracked per span:
  - agent.version
  - task.id
  - step.number
  - token.input / token.output
  - latency.ms
  - model.name
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports — OpenTelemetry is optional
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.debug("OpenTelemetry not installed — using local tracing only")


class AgentTelemetry:
    """OpenTelemetry-based distributed tracing for agent operations.

    Falls back to local console export if no OTLP collector is configured.
    """

    def __init__(self, service_name: str = "ai-agent", otlp_endpoint: str = ""):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.tracer = None
        self._spans: list[dict] = []   # Local fallback
        self._setup()

    def _setup(self):
        if not OTEL_AVAILABLE:
            logger.info("AgentTelemetry: using local span tracking (no OTLP)")
            return

        try:
            provider = TracerProvider()

            # Always add console exporter for local dev
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

            # Add OTLP exporter if endpoint configured
            if self.otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    otlp = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp))
                    logger.info("AgentTelemetry: OTLP export to %s", self.otlp_endpoint)
                except ImportError:
                    logger.warning("OTLP gRPC exporter not available — install opentelemetry-exporter-otlp")
                except Exception as e:
                    logger.warning("OTLP setup failed (collector may be offline): %s", e)

            from opentelemetry import trace
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(self.service_name)
            logger.info("AgentTelemetry: initialized for service '%s'", self.service_name)
        except Exception as e:
            logger.warning("AgentTelemetry initialization failed: %s", e)

    def trace_step(self, step_name: str, attributes: dict = None):
        """Decorator: trace an agent step function.

        Usage:
            @telemetry.trace_step("PLANNING")
            async def planning_step(self, ctx):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                t0 = time.time()
                attrs = (attributes or {}).copy()
                attrs.update({
                    "agent.service": self.service_name,
                    "agent.step": step_name,
                })

                if self.tracer:
                    with self.tracer.start_as_current_span(step_name, kind=SpanKind.INTERNAL) as span:
                        for k, v in attrs.items():
                            span.set_attribute(k, str(v)[:255])
                        try:
                            result = await func(*args, **kwargs)
                            span.set_attribute("latency.ms", (time.time() - t0) * 1000)
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)[:255]))
                            span.set_attribute("error", str(e)[:200])
                            raise
                else:
                    # Local fallback
                    try:
                        result = await func(*args, **kwargs)
                        self._spans.append({
                            "step": step_name,
                            "latency_ms": (time.time() - t0) * 1000,
                            "status": "ok",
                        })
                        return result
                    except Exception as e:
                        self._spans.append({
                            "step": step_name,
                            "latency_ms": (time.time() - t0) * 1000,
                            "status": "error",
                            "error": str(e)[:200],
                        })
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                t0 = time.time()
                if self.tracer:
                    with self.tracer.start_as_current_span(step_name, kind=SpanKind.INTERNAL) as span:
                        for k, v in (attributes or {}).items():
                            span.set_attribute(k, str(v)[:255])
                        try:
                            result = func(*args, **kwargs)
                            span.set_attribute("latency.ms", (time.time() - t0) * 1000)
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)[:255]))
                            raise
                else:
                    result = func(*args, **kwargs)
                    self._spans.append({"step": step_name, "latency_ms": (time.time() - t0) * 1000, "status": "ok"})
                    return result

            return wrapper if not isinstance(func, type) else sync_wrapper
        return decorator

    @contextmanager
    def span(self, name: str, **attributes):
        """Context manager for manual span creation."""
        t0 = time.time()
        if self.tracer:
            with self.tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v)[:255])
                yield span
                span.set_attribute("latency.ms", (time.time() - t0) * 1000)
        else:
            yield None
            self._spans.append({"step": name, "latency_ms": (time.time() - t0) * 1000, "status": "ok"})

    def get_spans(self) -> list[dict]:
        return self._spans[-50:]

    def status(self) -> dict:
        return {
            "otel_available": OTEL_AVAILABLE,
            "tracer_active": self.tracer is not None,
            "spans_local": len(self._spans),
            "otlp_endpoint": self.otlp_endpoint or "not configured",
        }
