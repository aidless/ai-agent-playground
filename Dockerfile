FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Dependencies layer (cached) ──────────────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Source code ──────────────────────────────────
COPY agent/         ./agent/
COPY ai_agent_playground/ ./ai_agent_playground/
COPY observability/ ./observability/
COPY scripts/       ./scripts/
COPY memory/        ./memory/
COPY skills/        ./skills/
COPY tests/         ./tests/
COPY resume_matcher/ ./resume_matcher/
COPY pyproject.toml uv.lock ./

# Create runtime directories
RUN mkdir -p /app/logs /app/sandbox_workspace /app/tenant_workspaces \
    /app/memory/audit_trails /app/memory/auto /app/memory/cost \
    /app/memory/reliability /app/skills/bootstrapped

EXPOSE 8000

ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uv", "run", "uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
