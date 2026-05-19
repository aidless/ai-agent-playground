# AI Agent Playground — Production-Grade Autonomous Agent System

[![Live Demo](https://img.shields.io/badge/demo-live-green)](http://47.98.106.182:8080)
[![Tests](https://img.shields.io/badge/tests-161%20passed-brightgreen)](https://github.com/aidless/ai-agent-playground/actions)
[![Security](https://img.shields.io/badge/security-14%2F14%20pentest-brightgreen)](scripts/pentest.py)
[![b3](https://img.shields.io/badge/b3%20benchmark-100%25-brightgreen)](scripts/b3_security_bench.py)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**Live Demo**: http://47.98.106.182:8080

A self-evolving AI agent system with 9 autonomous engines, 14/14 security penetration tests passed, and 100% b3 security benchmark compliance. Runs 24/7 on Alibaba Cloud.

---

## Benchmarks

| Metric | Score | Details |
|--------|-------|---------|
| Security Penetration | **14/14 (100%)** | 14 attack scenarios |
| b3 Security | **10/10 (100%)** | 5 categories (WDTA standard) |
| Code Repair (SWE-bench style) | **90% fix, 70% detect** | 10 real bug-fix tasks |
| Stress Test | **1000/1000, P95=150ms** | 50 concurrent |
| Self-Correction | **30%** | Feedback-driven retry |
| Test Suite | **161 passed, 0 failed** | Zero regressions |
| Availability | **10/10 subsystems healthy** | 24/7 monitoring |

## Architecture

9 autonomous engines coordinated by AutoPilot:

```
AutoPilot (autonomous coordinator)
├── AgentMatrix    — Multi-model routing (DeepSeek + Qwen2.5)
├── Debate         — Process-centric + competitive dual-mode
├── Evolution      — Perf tracking → template learning → optimize → rollback
├── Bootstrap      — Gap detection → code gen → AST validate → register
├── ReflectAction  — Tool failure → auto-degrade → substitute
├── MetaAgent      — Autonomous observe → decide → act
├── SelfPlay       — Generator → Solver → Evaluator curriculum loop
├── EvaluationGate — 3D quality (Interface + Functional + Utility)
└── UnifiedPipeline — Crew decompose → Debate → CrossReview → report
```

## Security

```
- 14/14 penetration test scenarios ✓
- 10/10 b3 benchmark attacks blocked ✓
- Prompt Injection detection (30+ patterns, CN+EN)
- Token brute-force rate limiting (5/min/IP)
- 256-bit HMAC-SHA256 token signatures
- Path traversal prevention (normPath + case-insensitive)
- API Key enforcement in production mode
- Intrusion detection (5 anomaly types)
- Audit log redaction (API keys, JWTs, Bearer tokens)
- Sandbox process isolation (terminate + kill)
```

## Quick Start

```bash
# Clone
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground

# Configure
cp .env.example .env
nano .env  # Add DEEPSEEK_API_KEY

# Run
uv sync
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8000
```

**Docker**:
```bash
docker-compose up -d
```

**Production deploy**: see [DEPLOY.md](DEPLOY.md)

## Run Benchmarks

```bash
uv run python scripts/pentest.py              # Security (14 scenarios)
uv run python scripts/b3_security_bench.py    # b3 benchmark (10 attacks)
uv run python scripts/code_bench.py           # Code repair (10 tasks)
uv run python scripts/multi_agent_bench.py    # Multi-agent (5 tasks)
uv run python scripts/stress_test.py          # Load test (1000 requests)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | DeepSeek V4 (primary), Qwen2.5:7b (reviewer) |
| Framework | FastAPI + AsyncIO + Uvicorn |
| Memory | ChromaDB + custom persistent store |
| Deployment | Docker + Alibaba Cloud ECS + systemd |
| Monitoring | Prometheus + CLEAR 5D panel |

## Project Structure

```
agent/            43 Python files (production engine)
├── async_core.py   Streaming agent with state machine
├── debate.py       Multi-model debate engine
├── evolution.py    Tool optimization + meta self-evolution
├── bootstrap.py    Skills auto-generation
├── sandbox_meta.py Sandboxed meta experimentation
├── self_play.py    Autonomous curriculum learning
└── server.py       30+ REST endpoints

scripts/           12 benchmark/deployment scripts
blog/              Technical blog posts (CN + EN)
tests/             161 test cases
```

## Blog Posts

- [中文：从学生项目到生产级 AI Agent](blog/from-student-to-production.md)
- [English: From Student Project to Production AI Agent](blog/from-student-to-production-en.md)

## Author

**Liu Zewen (刘泽文)** — B.Eng. Software Engineering 2026, Qilu Institute of Technology

GitHub: [@aidless](https://github.com/aidless) | Open to AI Application Developer positions
