# AI Agent Playground

Agent tutorials end at one prompt and one API call, which does not show whether
routing, safety checks, or evaluation actually run. This repository is that
stack, runnable offline, with the claims below checked against the tree.

## Status

[![CI](https://github.com/aidless/ai-agent-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/ai-agent-playground/actions/workflows/ci.yml)
[![mutation-control](https://github.com/aidless/ai-agent-playground/actions/workflows/mutation-control.yml/badge.svg)](https://github.com/aidless/ai-agent-playground/actions/workflows/mutation-control.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

<!-- ci-collect-only: 182 -->

The workflow badges are GitHub Actions status for `ci.yml` and `mutation-control.yml`. The Python badge is the language floor. There is no shield for a pentest rate, a repair rate, or a stress total. Those used to be static images. Why they came off: [FAILURE_MODES.md](FAILURE_MODES.md).

## Offline path

What you can run today. No hosted service and no model call.

```bash
git clone https://github.com/aidless/ai-agent-playground
cd ai-agent-playground
uv sync --locked
export DEEPSEEK_API_KEY=sk-ci-placeholder
export DEEPSEEK_BASE_URL=https://api.deepseek.com
uv run python -m pytest scripts/e2e_test.py tests --tb=short -q
uv run python scripts/pentest.py
```

`python -m pytest scripts/e2e_test.py tests` is the CI job (`.github/workflows/ci.yml`). The HTML comment above is the `--collect-only` count for that command, checked by `tests/test_claim_drift.py`. Do not quote a passed/failed total from this file. `tests/load_test.py` skips without locust, and `scripts/e2e_test.py::test_streaming_agent` skips on the placeholder key. A green Actions run is the pass/fail record.

`DEEPSEEK_*` is required at import by `agent/server.py` (the value must start with `sk-`). This selection does not send the placeholder to a model.

`scripts/pentest.py` is 14 in-process defense checks. It is not a third-party pentest and it is not part of CI.

To boot the API on your machine:

```bash
uv run uvicorn agent.server:app --host 127.0.0.1 --port 8000
```

`scripts/stress_test.py` then sends traffic at that process. It stores nothing in git.

## Historical cloud demo

An Alibaba Cloud ECS instance once ran this tree under systemd. That instance has been released. This repository does not publish a demo URL and it does not retain production telemetry. `blog/from-student-to-production.md`, `blog/from-student-to-production-en.md`, and `docs/process/` are period notes. Numbers in those files are not current evidence.

## What is in the tree

`agent/` is the service (routing, sandbox, identity, the self-check loops). `ai_agent_playground/` is the pipeline-style framework (`config`, `base`, `llm`). Four smaller packages (`hello_agent`, `code_review_agent`, `rag_qa_system`, `multi_agent_crew`) are the earlier portfolio pieces. Diagram: [ARCHITECTURE.md](ARCHITECTURE.md).

Scripts that call a live model (`scripts/code_bench.py`, `scripts/benchmark_engines.py`) are not CI and have no result file in git.

## For recruiters

The offline path above is the evidence. I am looking for AI application developer and AI engineer roles.
GitHub: [@aidless](https://github.com/aidless)

Liu Zewen (刘泽文), B.Eng. Software Engineering, Qilu Institute of Technology
