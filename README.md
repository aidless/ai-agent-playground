# AI Agent Playground

**A production-grade autonomous AI agent system — built from scratch, deployed 24/7, 161 tests, 0 failures.**

[![Live Demo](https://img.shields.io/badge/demo-live-green)](http://47.98.106.182:8080)
[![Tests](https://img.shields.io/badge/tests-161%20passed-brightgreen)](https://github.com/aidless/ai-agent-playground/actions)
[![Security](https://img.shields.io/badge/security-14%2F14%20pentest-brightgreen)](scripts/pentest.py)
[![Code Fix](https://img.shields.io/badge/code%20fix-90%25-brightgreen)](scripts/code_bench.py)
[![Stress Test](https://img.shields.io/badge/stress-1000%2F1000-blue)](scripts/stress_test.py)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

> **Live Demo**: http://47.98.106.182:8080 &nbsp;|&nbsp; **GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground)

---

## Why This Project Matters

Most AI Agent tutorials stop at "call an API, write a prompt." This project goes further — it's a **complete autonomous agent system** with 11 self-improving engines, automated security hardening, and production deployment. Built from zero by a 2026 graduate over one semester. 161 tests, 0 failures. Runs 24/7 on Alibaba Cloud.

**Key numbers:**

| Metric | Result |
|--------|--------|
| Security | 14/14 penetration tests passed, 10/10 b3 benchmark attacks blocked |
| Code Repair | 90% fix rate on real-world Python bugs |
| Load Test | 1000/1000 requests, P95=150ms, 50 concurrent |
| Test Suite | 161 passed, 0 failed — zero regressions |
| Autonomy | 11 engines: self-evolution, debate, bootstrap, meta-agent |
| Deployment | Alibaba Cloud ECS, systemd daemon, 24/7 uptime |

---

## What It Does

An autonomous agent system that doesn't just answer questions — it **improves itself**:

- **Detects its own capability gaps** → generates missing tools at runtime (Bootstrap Engine)
- **Evolves tool implementations** → optimizes code, rolls back on regression (Evolution Engine)
- **Debates with itself** → Primary proposes, Challenger critiques, Arbitrator synthesizes (Debate Engine)
- **Self-play training** → generates curriculum, improves via feedback loop (SelfPlay Engine)
- **Meta-agent oversight** → observes, decides, acts on system health (MetaAgent)

All 11 engines have real LLM-validated code, not just architecture diagrams.

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
Stress Test (50 concurrent):
  Total: 1000 | OK: 1000 (100%)
  Avg: 87ms | P50: 68ms | P95: 150ms | P99: 300ms

Code Repair (10 real bugs):
  Fix Rate:   90% (9/10)
  Detect Rate: 70% (7/10)
  Self-Correction: 30% (feedback-driven retry)
```

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
| Deployment | Docker + Alibaba Cloud ECS + systemd | 24/7 production grade |
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

## API Endpoints (OpenAI-Compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health — 10 subsystems |
| `/chat/completions` | POST | **OpenAI-compatible chat** (drop-in replacement) |
| `/autopilot/solve` | POST | Full 9-engine autonomous loop |
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
agent/              43 Python files (production engine)
├── async_core.py       Streaming agent + state machine
├── autopilot.py        Full 9-engine autonomous coordinator
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
tests/                161 test cases (0 failures)
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
GitHub: [@aidless](https://github.com/aidless) &nbsp;|&nbsp; Live Demo: http://47.98.106.182:8080

---

*Liu Zewen (刘泽文) — B.Eng. Software Engineering 2026, Qilu Institute of Technology*
