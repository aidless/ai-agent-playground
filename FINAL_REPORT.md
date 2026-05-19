# AI Agent Playground — Final System Report

**Author**: Liu Zewen (刘泽文) | B.Eng. Software Engineering 2026 | Qilu Institute of Technology
**GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground)
**Live**: http://47.98.106.182:8080 | **Tests**: 161 passed, 0 failed

---

## Executive Summary

A **production-grade autonomous AI agent system** built from scratch. Features 12 self-improving engines, passed 14/14 security penetration tests, achieved 100% on b3 security benchmark, and scored 90% on code repair tasks. Deployed 24/7 on Alibaba Cloud.

This is NOT a LangChain wrapper or API demo — every engine is custom-built with real LLM verification.

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │        AutoPilot (coordinator)   │
                    │  CLASSIFY→EXECUTE→VERIFY→IMPROVE │
                    └──────────────┬──────────────────┘
           ┌───────────────────────┼───────────────────────┐
    ┌──────▼──────┐  ┌───────────▼──────────┐  ┌─────────▼─────────┐
    │ AgentMatrix │  │     Debate Engine     │  │  Evolution Engine │
    │ Multi-model │  │ Process + Competitive │  │ Optimize→Rollback │
    │   Router    │  │    4-round debate     │  │  Template Learning│
    └─────────────┘  └───────────────────────┘  └───────────────────┘
    ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐
    │  Bootstrap  │  │    ReflectAction      │  │    MetaAgent      │
    │ Auto-gen    │  │  Degrade→Substitute   │  │ Observe→Decide    │
    │ AST valid   │  │  Failure detection    │  │     →Act          │
    └─────────────┘  └───────────────────────┘  └───────────────────┘
    ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐
    │  SelfPlay   │  │   Evaluation Gate     │  │Episodic Memory    │
    │ Curriculum  │  │  Interface+Func+Util  │  │ Store→Retrieve    │
    │  Learning   │  │   3D quality scoring  │  │  Reflexion paper  │
    └─────────────┘  └───────────────────────┘  └───────────────────┘
    ┌─────────────┐  ┌───────────────────────┐  ┌───────────────────┐
    │ Knowledge   │  │  Unified Pipeline     │  │ Sandbox Meta      │
    │ 133 papers  │  │ Crew→Debate→Review    │  │ Safe self-evolve  │
    │ Ollama RAG  │  │                       │  │ 161 test gate     │
    └─────────────┘  └───────────────────────┘  └───────────────────┘
```

---

## Benchmark Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **Security Penetration** | 14/14 (100%) | 14 attack scenarios, automated |
| **b3 Security (WDTA)** | 10/10 (100%) | 5 categories, all blocked |
| **Code Repair** | 90% fix, 70% detect | 10 real-world Python bugs |
| **Self-Correction** | 30% | Feedback-driven retry improvement |
| **Stress Test** | 1000/1000, P95=150ms | 50 concurrent, mixed endpoints |
| **Multi-Agent** | 7.7/10 | 5 collaboration tasks |
| **Pipeline Demo** | Single agent 7.5, 16427 chars | URL shortener design |

### Security Defense Matrix

| Defense | Mechanism |
|---------|-----------|
| Prompt Injection Guard | 38 patterns (CN+EN), all endpoints |
| Token Rate Limiting | 5/min/IP, auto-reset on success |
| HMAC-SHA256 Signatures | 256-bit, random salt per token |
| Path Traversal Prevention | `Path.resolve()` + case-insensitive |
| API Key Enforcement | Production mode: refuse startup without key |
| Intrusion Detection | 5 anomaly types, real-time alerts |
| Audit Log Redaction | Regex: API keys, JWTs, Bearer tokens |
| Sandbox Isolation | `multiprocessing.Process` + terminate→kill |
| Code Safety (AST) | Blocks os/subprocess/socket/ctypes imports |
| Memory Poison Detection | Statistical anomaly + LLM semantic verify |
| Goal Drift Detection | Keyword overlap tracking, 3-step threshold |
| Cost Tracking | Session-level, real-time budget warnings |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | DeepSeek V4 (primary, Anthropic protocol), Qwen2.5:7b (reviewer, Ollama local) |
| **Framework** | FastAPI + AsyncIO + Uvicorn (4 workers) |
| **Validation** | Pydantic v2 (max_length, model_validator) |
| **Streaming** | SSE (Server-Sent Events, `/v1/chat/stream`) |
| **Vector DB** | ChromaDB (133 papers, 3584-dim Ollama embeddings) |
| **Sandbox** | multiprocessing.Process isolation |
| **Auth** | HMAC-SHA256 tokens + API Key middleware |
| **Deployment** | Docker + systemd + Alibaba Cloud ECS (2C4G) |
| **Monitoring** | Prometheus + OpenTelemetry + CLEAR panel |
| **Testing** | pytest (161 tests) + custom pentest + b3 + code bench |
| **Docs** | README + ARCHITECTURE.md + DEPLOY.md + CHANGELOG + blog (CN/EN) |

---

## API Surface

35+ REST endpoints:

| Group | Endpoints |
|-------|-----------|
| Core | `/chat/completions`, `/v1/chat/stream`, `/health` |
| SuperAgent | `/super/debate`, `/super/evolve`, `/super/status`, `/super/degrade`, `/super/meta/experiment` |
| AutoPilot | `/autopilot/solve`, `/autopilot/status` |
| SelfPlay | `/selfplay/train`, `/selfplay/status` |
| Evaluation | `/eval/gate`, `/eval/ab`, `/eval/stats` |
| Knowledge | `/knowledge/collect`, `/knowledge/query`, `/knowledge/search`, `/knowledge/status` |
| Matrix | `/matrix/solve`, `/matrix/status` |
| Security | `/security/intrusion`, `/ciso/approval`, `/ciso/pending` |
| UI | `/`, `/dashboard`, `/chat`, `/portfolio` |

---

## Key Innovations

### 1. Self-Evolving Tools
The Evolution Engine tracks tool performance and automatically optimizes underperforming tools via LLM-generated code patches. Old versions are stored as rollback snapshots. **Verified**: bubble sort evolved to Timsort, 13-line diff, validated and registered.

### 2. Skills Bootstrapping
When the agent detects a capability gap ("I need X but don't have it"), it generates new tool code, validates syntax + AST safety, and registers the tool for immediate use. **Verified**: DeepSeek generated a 1043-char markdown-to-JSON parser, correctly output structured data.

### 3. Sandboxed Meta Self-Evolution
The MetaAgent can safely experiment on copies of its own source code. Changes are applied in an isolated sandbox, the full test suite runs, and only passing proposals are saved for human review. **Verified**: uptime.py successfully evolved, 161/161 tests passed.

### 4. Knowledge-Driven Decision Making
The agent can query 133 indexed AI research papers via semantic search (Ollama embeddings + ChromaDB) to answer questions with research-backed evidence. **Verified**: RAG queries return cited answers referencing specific papers.

### 5. Process-Centric Multi-Model Debate
Primary model proposes step-by-step → Challenger critiques each step ([CORRECT]/[GAP]/[ERROR]) → Synthesis produces improved answer. Collaborative, not adversarial. **Verified**: 4-round debate on rate limiting design, 27K+ chars of structured argumentation.

---

## Deployment

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
./deploy.sh setup && nano .env && ./deploy.sh start
```

3 minutes from zero to running. Docker and bare-metal both supported. Full guide: [DEPLOY.md](DEPLOY.md)

---

## Project Stats

| Metric | Value |
|--------|-------|
| Python modules | 65+ |
| Test cases | 161 |
| API endpoints | 35+ |
| Engines | 12 autonomous |
| Security vulns fixed | 14 → 0 |
| Lines of code | ~20,000+ |
| Commits | 25+ |
| Docker images | 1 (multi-service) |
| Cloud cost | ~¥5.50/day (¥0.23/hr) |

---

*Built by Liu Zewen, 2026 graduate seeking AI Application Developer roles.*
*Contact: GitHub [@aidless](https://github.com/aidless)*
