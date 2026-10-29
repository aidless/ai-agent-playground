# AI Agent Playground

**A self-improving autonomous AI agent system built from scratch; full CI: 166 passed, 1 skipped, 0 failed.**

[![Tests](https://img.shields.io/badge/tests-166%20passed-brightgreen)](https://github.com/aidless/ai-agent-playground/actions)
[![Security](https://img.shields.io/badge/security-14%2F14%20pentest-brightgreen)](scripts/pentest.py)
[![Code Fix](https://img.shields.io/badge/code%20fix-90%25-brightgreen)](scripts/code_bench.py)
[![Stress Test](https://img.shields.io/badge/stress-1000%2F1000-blue)](scripts/stress_test.py)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

> **Status note (2026-10):** this system previously ran on an Alibaba Cloud ECS
> instance under systemd. **That instance has been released, so there is no live
> demo URL and no retained production telemetry.** A few documents written while
> it was live (`blog/from-student-to-production*.md`, `PAPER.md`,
> `FINAL_REPORT.md`) carry the former URL and now say so at the top. This README
> presents reproducible local checks, not current production evidence.

---

## Why This Project Matters

Most AI Agent tutorials stop at "call an API, write a prompt." This project
goes further: it includes **five self-improvement engines** across **55 agent
modules**, automated security checks, and a service/orchestration layer. The
full CI suite reports 166 passed and 1 skipped; the narrower `tests/` command
reports 161 passed and 1 skipped.

**Key numbers:**

| Metric | Result |
|--------|--------|
| Security | 14/14 penetration tests passed, 10/10 b3 benchmark attacks blocked |
| Code Repair | 90% fix rate on real-world Python bugs |
| Local stress script | 1000/1000 synthetic local requests; see `scripts/stress_test.py` |
| Full CI suite | 166 passed, 1 skipped, 0 failed |
| `tests/` only | 161 passed, 1 skipped, 0 failed |
| Autonomy | 5 self-improvement engines: evolution, debate, bootstrap, self-play, meta-agent |
| Deployment | Previous Alibaba Cloud ECS + systemd deployment; instance released, no retained telemetry |

---

## What It Does

An autonomous agent system that doesn't just answer questions — it **improves itself**:

- **Detects its own capability gaps** → generates missing tools at runtime (Bootstrap Engine)
- **Evolves tool implementations** → optimizes code, rolls back on regression (Evolution Engine)
- **Debates with itself** → Primary proposes, Challenger critiques, Arbitrator synthesizes (Debate Engine)
- **Self-play training** → generates curriculum, improves via feedback loop (SelfPlay Engine)
- **Meta-agent oversight** → observes, decides, acts on system health (MetaAgent)

These 5 are the self-improvement engines; the whole system is **55 modules** under `agent/`
(memory, cost tracking, governance, sandbox, reliability, and others).
`scripts/benchmark_engines.py` measures the three that are directly comparable —
baseline vs Debate vs Matrix.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    AutoPilot Loop                       │
│                                                          │
│  CLASSIFY → EXECUTE → VERIFY → REFLECT → IMPROVE → RETRY
│      │                              │                    │
│      ▼                              ▼                    │
│  Matrix Router              ┌───────┴───────┐           │
│  (role + model)             │  4 Strategies  │           │
│                             ├── Debate       │           │
│                             ├── Evolution    │           │
│                             ├── Bootstrap    │           │
│                             └── Meta Observe │           │
│                             └───────┬───────┘           │
│                                     ▼                    │
│                              Evaluation Gate             │
│                              (I+F+U scoring)             │
└──────────────────────────────────────────────────────────┘
```

**Agent State Machine:** `IDLE → PLANNING → TOOL_CALL → REFLECT → LEARN → DONE`

Detailed architecture: see [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Security: From 12 Vulnerabilities to 14/14

I audited my own code against OWASP Top 10 for LLM Applications and found **12 critical/high vulnerabilities** — then fixed every one:

| # | Vulnerability | Fix |
|---|---|---|
| 1 | Sandbox timeout bypassable (thread join) | `multiprocessing.Process` + `terminate()`/`kill()` |
| 2 | Path traversal via `..` / Unicode / case | `Path.resolve()` + case-insensitive match |
| 3 | API Key auth silently disabled when unset | `production` mode refuses startup without key |
| 4 | Token entropy 64-bit (truncation) | HMAC-SHA256, 64-char hex, random salt |
| 5 | CORS `allow_origins=["*"]` with credentials | Explicit origin whitelist |
| 6 | Token brute-force (no rate limit) | 5 attempts/min/IP, counter resets on success |
| 7 | Prompt injection unprotected | 30+ pattern guard (CN + EN), all endpoints |
| 8 | Identity creator not tracked | Audit trail: who created which identity |
| 9 | Audit log redaction incomplete | Regex: API keys, JWTs, Bearer tokens |
| 10 | Coarse-grained permissions (4 roles) | Resource-level `PermissionGrant` with `fnmatch` |
| 11 | Tenant ID spoofable (Header) | Cryptographic tenant binding |
| 12 | Tool risk misclassified | 4-level risk model with CISO approval gate |

**Result:** 14/14 automated penetration tests pass. Run `uv run python scripts/pentest.py` to verify.

---

## Benchmarks

### Security
```
Prompt Injection      ████████████████████ 100% blocked
Path Traversal        ████████████████████ 100% blocked
Token Brute Force    ████████████████████ 100% blocked
Bootstrap Safety      ████████████████████ 100% blocked
Evolution Safety      ████████████████████ 100% blocked
```

### Performance
```
Local stress script (50 concurrent, synthetic):
  Total: 1000 | OK: 1000 (100%)

Code Repair (10 real bugs):
  Fix Rate:   90% (9/10)
  Detect Rate: 70% (7/10)
  Self-Correction: 30% (feedback-driven retry)
```

The historical production throughput and P95 figures are intentionally omitted:
the deployment instance and its telemetry were not retained, so they cannot be
verified from this repository.

### Engine Comparison
| Engine | Avg Score | Latency |
|--------|-----------|---------|
| Baseline (DeepSeek V4 only) | 8.9/10 | ~13s |
| Debate (process-centric) | 8.3/10 | ~119s |
| Matrix (multi-model route) | 8.9/10 | ~30s |

**Insight:** DeepSeek V4 is already strong. Debate doesn't help simple tasks but fixes 1/5 baseline errors on hard code-debugging tasks. Key: **selective use**, not blind debate for every request.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-------------|-----|
| LLM | DeepSeek V4 (primary), Qwen2.5:7b (reviewer) | OpenAI-compatible, model-agnostic |
| Framework | FastAPI + AsyncIO + Uvicorn | Standard Python AI serving stack |
| Vector DB | ChromaDB + all-MiniLM-L6-v2 | Lightweight RAG, easy to deploy |
| Security | Process sandbox + AST scan + HMAC auth | Defense in depth |
| Deployment | Docker + Alibaba Cloud ECS + systemd | production-grade config
| Monitoring | Prometheus + CLEAR 5D panel | Cost, Latency, Efficacy, Assurance, Reliability |

---

## Quick Start

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env
# Edit .env: add DEEPSEEK_API_KEY
uv sync
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8000
```

**Docker:** `docker-compose up -d` &nbsp;|&nbsp; **Production:** `./deploy.sh setup && nano .env && ./deploy.sh start`

---

## Running the Tests

The full CI suite runs `scripts/e2e_test.py` plus `tests/` and reports
**166 passed, 1 skipped**. Running only `tests/` reports **161 passed,
1 skipped**. Both are fully offline - no API calls are made.

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground

# --group dev is required: pytest lives in the dev dependency group
uv sync --group dev

# agent/server.py validates the key's format (must start with `sk-`) at import
# time; the value is never used to reach the network during tests.
export DEEPSEEK_API_KEY="sk-local-test-dummy"

uv run --group dev pytest tests/ -q
# 161 passed, 1 skipped
```

To reproduce the full CI test count:

```bash
uv run --group dev pytest scripts/e2e_test.py tests/ -q
# 166 passed, 1 skipped
```

The skipped test is the locust load test (`tests/load_test.py`); add it with
`uv add --dev locust`.

If `pytest` resolves to a global install instead of the venv, prefix with
`uv run --group dev` as above — plain `uv run pytest` picks up whatever
`pytest` is on PATH.

Note for contributors: `prometheus-client`, `sse-starlette`, and `tenacity`
are imported by the code and the tests but were missing from `dependencies`;
`pytest` and `pytest-asyncio` were missing entirely (now in `[dependency-groups].dev`).

---

## API Endpoints (OpenAI-Compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health — 10 subsystems |
| `/chat/completions` | POST | **OpenAI-compatible chat** (drop-in replacement) |
| `/autopilot/solve` | POST | Full autonomous loop |
| `/super/debate` | POST | Multi-model debate |
| `/super/evolve` | POST | Tool evolution (optimize → rollback) |
| `/super/meta/experiment` | POST | Sandboxed self-evolution |
| `/eval/gate` | POST | 3D quality evaluation (I+F+U) |
| `/eval/ab` | POST | A/B testing |
| `/matrix/solve` | POST | Multi-agent routing |
| `/security/intrusion` | GET | Intrusion detection status |

---

## Project Structure

```
agent/              55 Python files (agent engine)
├── async_core.py       Streaming agent + state machine
├── autopilot.py        Full autonomous coordinator
├── debate.py           Multi-model debate
├── evolution.py        Tool optimization + template learning
├── bootstrap.py        Runtime tool generation + AST safety
├── reflect_action.py   Failure detection + auto-degradation
├── self_play.py        Curriculum learning self-improvement
├── sandbox_meta.py     Safe self-modification sandbox
├── matrix.py           Multi-model routing
├── eval_gate.py        3D quality evaluation gate
├── intrusion.py        Anomaly detection (5 types)
├── identity.py         RBAC + session tokens + rate limiting
└── server.py           30+ REST endpoints

ai_agent_playground/  23 files (framework layer, HuggingFace-inspired)
tests/                162 test cases (161 passed, 1 skipped)
scripts/e2e_test.py  5 additional tests used by the full CI run
scripts/              22 benchmark & deployment scripts
blog/                 Technical blog (CN + EN)
```

---

## Blog

- [中文：从学生项目到生产级 AI Agent](blog/from-student-to-production.md)
- [English: From Student Project to Production AI Agent](blog/from-student-to-production-en.md)

---

## For Recruiters

This project demonstrates:

1. **Real production engineering** — not a tutorial clone, but a system designed, built, deployed, and maintained by one person
2. **Security mindset** — found and fixed 12 vulnerabilities, automated penetration testing
3. **Self-directed learning** — read 20+ AI research papers, applied insights to implementation
4. **System-level thinking** — architecture, observability, deployment, not just model calling

**I'm open to AI Application Developer / AI Engineer roles.**  
GitHub: [@aidless](https://github.com/aidless)

---

*Liu Zewen (刘泽文) — B.Eng. Software Engineering, Qilu Institute of Technology*
