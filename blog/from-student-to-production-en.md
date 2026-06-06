# From Student Project to Production AI Agent: A Graduate's Technical Retrospective

> **Live Demo**: http://47.98.106.182:8080 &nbsp;|&nbsp; **GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground) &nbsp;|&nbsp; **161 tests, 0 failures**

---

## One-Line Summary

I built a **production-grade autonomous AI agent system** from scratch over one semester — 9 self-improving engines, 14/14 penetration tests passed, deployed 24/7 on Alibaba Cloud. This is not a tutorial follow-along. Every architecture decision was mine to make, mine to debug, and mine to fix.

---

## Why I'm Writing This

When I started learning AI Agent development last year, I hit a weird gap:

- **Tutorial level**: Call an API, write a prompt, run it, done.
- **Production level**: Almost nothing. Everyone says "Agent will work autonomously" — but no one explains "how to keep it safe on a server running for 3 months without crashing."

So I decided: **build one myself, step on every landmine, and write down what I learned.**

This retrospectives is not a tutorial. It's the real architecture decisions, the bugs I hit, and the data I collected.

---

## Phase 1: From Pipeline to 9-Engine Autonomy

### The Original Design

Inspired by HuggingFace Transformers source code, the first Agent was a simple Pipeline:

```
preprocess → _forward → postprocess
```

`preprocess` turns user input into model-readable format. `_forward` calls the LLM. `postprocess` returns output to user. Clean. Also insufficient.

### The Problems

Once the Agent was running, two fundamental issues emerged:

1. **Agents fail** — tool calls error out, logic goes in circles, loops never exit. Pipeline has no "reflection" or "correction" step.
2. **Agents should improve** — why make the same mistake twice? Pipeline has no "learning" step.

### The Current Architecture

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

**Key**: This is not an architecture diagram — every engine has code, tests, and real LLM-verified behavior.

---

## Phase 2: Security Audit — From 12 Vulnerabilities to 14/14

### I Thought I Was Secure

The system was running, features were working. I felt good. Then I seriously audited my own code against **OWASP Top 10 for LLM Applications**.

**Result: 12 Critical/High vulnerabilities.**

| # | Vulnerability | Impact |
|---|---|---|
| 1 | Sandbox timeout bypassable (thread join) | Attacker can run malicious code forever |
| 2 | Path traversal via `..` / Unicode / case | Read any file on server |
| 3 | API Key auth silently disabled when unset | Production exposed with no auth |
| 4 | Token signature entropy 16 bytes | GPU can brute-force |
| 5 | CORS `*` + credentials | Any website can send authenticated requests |
| 6 | No token brute-force rate limit | Infinite attempts |
| 7 | Prompt Injection unprotected | User can inject instructions to hijack Agent |
| 8 | Identity creator not tracked | Cannot trace who created which identity |
| 9 | Audit log redaction incomplete | API Keys may leak to logs |
| 10 | Coarse-grained permissions (4 roles) | No resource-level access control |
| 11 | Tenant isolation bypassable (Header forgery) | Spoof Tenant ID |
| 12 | Misclassified tool risk levels | `run_python` marked medium, should be high |

### The Fix Process

I didn't fix them one by one. I **wrote an automated penetration test script** `scripts/pentest.py` that simulates 14 attack scenarios. Fix one vulnerability, re-run all tests, ensure no regressions.

**After fix:**

```
SECURITY PENETRATION TEST — 14 attack scenarios
  [PASS] 1. Prompt injection blocked
  [PASS] 2. Legitimate message allowed
  [PASS] 3. Path traversal blocked
  [PASS] 4. Case-insensitive path blocked
  [PASS] 5. Token rate limiting (5/min/IP)
  [PASS] 6. Token signature entropy (64-char hex)
  [PASS] 7. Tool auto-degradation works
  [PASS] 8. Bootstrap safety blocks unsafe imports
  [PASS] 9. Audit log redacts API keys
  [PASS] 10. Resource-level permissions enforced
  [PASS] 11. Bootstrap validates safe code
  [PASS] 12. Evolution blocks dangerous code
  [PASS] 13. API key production enforcement
  [PASS] 14. Intrusion detection triggers

RESULTS: 14/14 defenses passed — penetration-test ready
```

**These 14 tests aren't run manually — `uv run python scripts/pentest.py` runs them all. After any change, re-run to ensure no regression.**

---

## Phase 3: Making the Agent Actually "Self-Evolve"

Many articles talk about SuperAgent being "autonomously evolving AI" — but few give you runnable code. My three engines all have real LLM verification.

### 1. Evolution Engine (verified with DeepSeek V4)

**Scenario**: DeepSeek V4 calls a sort tool that uses O(n²) bubble sort. After 3 consecutive failures, the system auto-triggers evolution.

**Evolution process:**
1. LLM reads tool source + performance metrics + error history
2. Uses template learning — references past successful optimizations
3. LLM generates optimized version
4. AST safety check + `compile()`
5. Atomically replaces old version; stores rollback snapshot

**Real output:**
```diff
--- sort_numbers_v0 (bubble sort, O(n²))
+++ sort_numbers_v1 (Timsort, O(n log n))
    def sort_numbers(params: dict) -> str:
-       for i in range(len(xs)):
-           for j in range(len(xs)-1):
-               if xs[j] > xs[j+1]:
-                   xs[j], xs[j+1] = xs[j+1], xs[j]
+       xs.sort()  # Python's Timsort
```

Before: 200ms P95. After: 12ms P95. **17x improvement.**

### 2. Bootstrap Engine

**Scenario**: Agent detects in reflection: "I need to parse Markdown tables but don't have this tool."

**Bootstrap process:**
1. LLM generates tool code (1,043-char `markdown_table_to_json`)
2. `compile()` syntax check
3. AST safety scan (blocks `import os/subprocess/socket`)
4. Register to ToolRegistry
5. **Immediately available** — next loop can call this new tool

**Real verification**: DeepSeek-generated tool executed correctly, output:
```json
[{"Name": "Alice", "Age": "25"}, {"Name": "Bob", "Age": "30"}]
```

### 3. Sandbox Meta Evolution (HYPERAGENTS Paper Implemented)

This is the Meta/HYPERAGENTS self-referential loop made practical:

1. Copy `agent/` source to sandbox directory
2. LLM reads its own source code, proposes improvements
3. Apply changes to sandbox copy
4. Run full test suite (161 tests)
5. **161/161 pass** → save as human-reviewed proposal; **any fail** → destroy sandbox, log error

**This experiment actually ran** — `agent/uptime.py` was successfully evolved in sandbox. 161/161 tests passed.

---

## Phase 4: Engineering — From Laptop to Cloud

### Stress Test

```
Phase 1: 50 concurrent to /health...
  /health: 500/500 OK | avg=103ms p50=100ms p95=255ms

Phase 2: 50 concurrent mixed endpoints...
  Mixed: 500/500 OK | avg=71ms p50=68ms p95=117ms

STRESS TEST RESULTS (50 concurrent)
  Total: 1000 | OK: 1000 (100%)
  Avg: 87ms | P50: 68ms | P95: 150ms | P99: 300ms
  P1 target: p95<=3000ms PASS
  P2 target: p99<=5000ms PASS
```

### Deployment

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
./deploy.sh setup && nano .env && ./deploy.sh start
```

Running on **Alibaba Cloud ECS (2C4G, ~$0.03/hour)**. Live at http://47.98.106.182:8080

---

## Benchmarks: Data-Driven Decisions

5 questions across coding, reasoning, security, design, algorithms:

| Engine | Avg Score | Latency |
|--------|-----------|---------|
| Baseline (DeepSeek V4 only) | 8.9/10 | ~13s |
| Debate (process-centric) | 8.3/10 | ~119s |
| Matrix (multi-agent routing) | 8.9/10 | ~30s |

**Finding**: DeepSeek V4 baseline is already strong. Debate doesn't help simple tasks but fixes **1/5 baseline errors on hard code-debugging tasks**.

**Key insight**: It's not "debate is always better than single model" — it's **selective activation**. Don't blindly run debate for every request. Only enable it when baseline confidence is low / task is complex.

---

## The Numbers

| Metric | Value |
|--------|-------|
| Tests | 161 passed, 0 failed |
| Security vulns found → fixed | 12 → 0 |
| Penetration tests | 14/14 (100%) |
| b3 security benchmark | 10/10 (100%) |
| Code repair | 90% fix rate, 70% detection |
| Self-correction | 30% (feedback-driven retry) |
| Engines | 9 autonomous |
| API endpoints | 30+ REST |
| Python modules | 50+ |
| Stress test | 1000/1000, P95=150ms |
| Deployment | Alibaba Cloud ECS, 24/7 online |

---

## 5 Things I Learned

### 1. Students can build production systems

The prerequisite isn't "having a big team" — it's "reading source code, reading papers, and writing your own tests."

The most valuable thing in this project isn't the code — it's the **20+ AI research papers I read** and applied: HyperAgents' meta-evolution, Debate's adversarial verification, Self-Play's curriculum learning. All landed as runnable code.

### 2. Security isn't an add-on

I should have done the security audit on day 1, not after the system was "feature-complete." My first deployable version had 12 critical vulnerabilities — if I'd audited earlier, I'd have written safer code from the start.

**Lesson**: Penetration tests must be automated. `scripts/pentest.py` is the highest-ROI script in this project.

### 3. AI Agent core isn't the prompt

It's the **governance, evolution, evaluation, and rollback engineering loop**.

Prompt determines what the Agent can do. Engineering determines whether the Agent **keeps doing the right thing consistently**. Self-evolution without rollback is a disaster waiting to happen.

### 4. Selectivity beats comprehensiveness

Not every request needs debate. Not every tool needs evolution. Not every capability gap needs a new bootstrapped tool.

**A system needs to learn when "not to do something"** — that's harder than "what to do," and more valuable.

### 5. Engineering judgment > tech stack depth

I've interviewed at places where candidates say "I know LangChain / LlamaIndex." The real question: **what problem did you solve with it? What were the bottlenecks? How did you trade off?**

This project tries to show not "I used DeepSeek V4" but **"under what circumstances I chose DeepSeek V4 over other options, and what that choice bought me."**

---

## If You're Also Building AI Agents

The project is fully open source. You can:

- **Run it**: `git clone` + `uv sync` + add an API key. Running in 5 minutes.
- **Read the code**: `agent/` has 43 production-grade files. Every engine has tests.
- **Open an Issue**: Found a bug or have an idea? GitHub Issues welcome.

**GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground) &nbsp;|&nbsp; **Live Demo**: http://47.98.106.182:8080

---

*Liu Zewen | B.Eng. Software Engineering 2026 | Qilu Institute of Technology | Open to AI Application Developer roles*
