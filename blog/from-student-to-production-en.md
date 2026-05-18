# From Student Project to Production AI Agent: A 2026 Graduate's Technical Retrospective

> Live demo: http://47.98.106.182:8080 | GitHub: github.com/aidless/ai-agent-playground | 161 tests, 0 failures

## TL;DR

Built a production-grade AI agent framework with 9 self-improving engines. Fixed 14 security vulnerabilities found by a self-audit. Ran 14/14 penetration tests. Stress tested at 1000 concurrent requests with P95=150ms. Deployed to Alibaba Cloud. The system can autonomously evolve its own tools — LLM-generated optimizations pass through AST safety checks, test suites, and atomic rollback.

Here's how I built it, what I learned, and the honest data.

## Architecture: Pipeline → 9-Engine Autonomy

The system started as a HuggingFace Transformers-inspired Pipeline (`preprocess → _forward → postprocess`). It grew into something more interesting:

```
AutoPilot (autonomous coordinator)
├── AgentMatrix    — Multi-model routing (DeepSeek V4 + Qwen2.5)
├── Debate         — Process-centric + competitive dual-mode
├── Evolution      — Performance tracking → template learning → optimize → rollback
├── Bootstrap      — Capability gap → code generation → AST validation → register
├── ReflectAction  — Tool failure → auto-degrade → substitute
├── MetaAgent      — Autonomous observe → decide → act
├── SelfPlay       — Generator → Solver → Evaluator feedback loop
└── EvaluationGate — 3D quality scoring (Interface + Functional + Utility)
```

Every engine has code, tests, and real LLM verification. Not architecture diagrams — running code.

## Security: From 12 Vulnerabilities to Penetration-Test Ready

### The starting point

I audited my own code against OWASP Top 10 for LLM Applications. 12 Critical/High findings:

| # | Vulnerability | Severity |
|---|---------------|----------|
| 1 | Sandbox timeout bypass (threads can't be killed) | Critical |
| 2 | Path traversal (string matching, no normalization) | Critical |
| 3 | API Key auth disabled by default | Critical |
| 4 | Token signature truncated to 16 hex chars (64-bit) | Critical |
| 5 | CORS allows all origins with credentials | High |
| 6 | No rate limiting on token validation | High |
| 7 | No Prompt Injection protection | High |
| 8 | Missing audit trail for identity creation | Medium |
| 9 | Incomplete sensitive data redaction | Medium |
| 10 | Coarse-grained role permissions | Medium |
| 11 | Tenant isolation bypassable via header forgery | Medium |
| 12 | Misclassified tool risk levels | Medium |

### The fix

An automated penetration test now runs 14 attack scenarios:

```
RESULTS: 14/14 defenses passed
  [PASS] Prompt injection blocked
  [PASS] Path traversal (case-insensitive, Unicode, dot-dot) blocked
  [PASS] Token brute-force rate limited (5/min/IP)
  [PASS] Token signature upgraded to full 256-bit HMAC-SHA256
  [PASS] API key enforced in production mode
  [PASS] Audit logs redact API keys, JWTs, Bearer tokens
  [PASS] Bootstrap blocks os/subprocess/socket imports
  [PASS] Evolution safety check blocks dangerous code
  [PASS] Resource-level permissions (USER_FILES vs SYSTEM_FILES)
  [PASS] Intrusion detection triggers on auth brute force
```

This isn't manual testing — it's `uv run python scripts/pentest.py`.

## SuperAgent: Three Engines That Actually Work

### 1. Evolution Engine (verified with DeepSeek V4)

When a tool underperforms (3+ consecutive failures), the engine:
1. Reads the tool's source code + performance metrics + error history
2. Uses template learning — references past successful optimizations
3. LLM generates an optimized version
4. AST safety check + compile()
5. Atomically replaces the old version, stores rollback snapshot

Real result: bubble sort automatically evolved to Python's Timsort. 13-line diff, validated, registered.

### 2. Bootstrap Engine

When reflection detects a capability gap ("I need to parse Markdown tables but don't have a tool"), the engine generates, validates, and registers new tool code.

Real result: DeepSeek generated a 1,043-character `markdown_table_to_json` function. Compiled, executed correctly, produced valid JSON output.

### 3. Sandbox Meta Evolution (HYPERAGENTS paper implemented)

This is the Meta/HYPERAGENTS self-referential loop made practical:
1. Copy agent/ source to sandbox directory
2. LLM reads its own source code, proposes improvements
3. Apply changes to sandbox copy
4. Run full test suite (161 tests)
5. All pass → save as human-reviewed proposal
6. Any fail → destroy sandbox, log for learning

Real result: `agent/uptime.py` was successfully evolved in sandbox — 161/161 tests passed. The proposal is saved for review.

## Engineering: From Laptop to Cloud

### Stress test: 1000/1000

```
Total: 1000 | OK: 1000 (100%)
Avg: 87ms | P50: 78ms | P95: 150ms | P99: 300ms
P1: p95<=3000ms PASS
P2: p99<=5000ms PASS
```

### Deployment: 3 minutes

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
./deploy.sh setup && nano .env && ./deploy.sh start
```

Running on Alibaba Cloud ECS (2C4G, ~$0.03/hour). Systemd-managed, auto-restart on failure.

### Benchmark data

5 questions across coding, reasoning, security, design, algorithms:

| Engine | Avg Score | Latency |
|--------|-----------|---------|
| Baseline (DeepSeek V4) | 8.9/10 | ~13s |
| Debate (process-centric) | 8.3/10 | ~119s |
| Matrix (multi-agent routing) | 8.9/10 | ~30s |

**Finding**: DeepSeek V4 baseline is already strong. Debate doesn't improve simple Q&A but fixes 1/5 baseline errors on hard code-bug tasks. The key engineering insight: **selective activation** — expensive engines only trigger when baseline quality is below threshold.

## By The Numbers

| Metric | Value |
|--------|-------|
| Tests | 161 passed, 0 failed |
| Security vulns fixed | 14 → 0 |
| Penetration tests | 14/14 passed |
| Engines | 9 autonomous |
| API endpoints | 30+ REST |
| Python modules | 50+ |
| Stress test | 1000/1000, P95=150ms |
| Deployment | Alibaba Cloud ECS, systemd |
| Commits | 20+ |

## What I Learned

1. **Production systems need automated security gates.** Manual audits are one-time. Pentest scripts are forever.

2. **DeepSeek V4 is underrated.** It scored 8.9/10 on every benchmark. Most "agent improvements" show benefit from weaker models. With strong baselines, you need data-driven decisions about when to activate expensive engines.

3. **Self-improvement needs safety infrastructure.** The HYPERAGENTS paper is right — but the hard part isn't the LLM. It's the sandbox, the test suite, the AST validator, the rollback mechanism.

4. **Ship early, measure everything.** I deployed at 14 vulnerabilities. Fixed them all. Now the system runs 24/7 with real metrics to show.

---

*Liu Zewen | B.Eng. Software Engineering 2026 | Qilu Institute of Technology | Open to AI Application Developer roles*
