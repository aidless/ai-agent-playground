# ARCHITECTURE.md — AI Agent Playground

## Overview

A production-grade AI agent framework with 5 projects, designed as a job-hunting
portfolio. Built from scratch with Python, FastAPI, DeepSeek API, and ChromaDB.

```
133 tests  |  0 failures  |  Security Score 8.5/10
```

## Layer Stack (bottom-up)

```
┌──────────────────────────────────────────────────┐
│  Endpoints  │  server.py  (30+ REST endpoints)   │
├──────────────────────────────────────────────────┤
│  SuperAgent │  debate.py  reflect_action.py      │
│             │  bootstrap.py                      │
├──────────────────────────────────────────────────┤
│  Orchestration │  orchestrator.py  crew_agent.py │
│                │  message_bus.py  attn_router.py │
├──────────────────────────────────────────────────┤
│  Agent Core │  async_core.py  state.py           │
│             │  llm_client.py  async_llm_client.py│
├──────────────────────────────────────────────────┤
│  Security   │  security.py  identity.py  tenancy │
│             │  sandbox.py  intrusion.py  ciso    │
├──────────────────────────────────────────────────┤
│  Governance │  governance.py  alerting.py        │
│             │  cost_tracker.py  reliability.py   │
├──────────────────────────────────────────────────┤
│  Memory     │  memory.py  auto_memory.py         │
│             │  context_compressor.py  skills.py  │
├──────────────────────────────────────────────────┤
│  Deployment │  deploy.py  blue_green.py          │
│             │  root_cause.py  uptime.py          │
├──────────────────────────────────────────────────┤
│  Obs        │  tracer.py  clear_metrics.py       │
│             │  monitoring.py  portfolio.py       │
└──────────────────────────────────────────────────┘
```

## Agent Loop (the core state machine)

```
  IDLE ──▶ PLANNING ──▶ TOOL_CALL ──▶ REFLECT ──▶ LEARN ──▶ DONE
              ▲              │            │          │
              └──────────────┘            │          │
                                          ▼          ▼
                                     ReflectAction  Bootstrap
                                     (degrade       (generate
                                      tools)         new tools)
```

Defined in `agent/state.py` as `AgentState` enum. The loop lives in
`agent/async_core.py::AsyncAgent.run_stream()`.

Each cycle:
1. **PLANNING** — LLM decides what to do (text response or tool calls)
2. **TOOL_CALL** — Concurrent execution via sandbox + governance wrapping
3. **REFLECT** — LLM self-critique; triggers ReflectAction engine
4. **LEARN** — Extract lessons to memory; triggers Bootstrap engine if gap detected

## Directory Map

### `agent/` — Production engine (43 modules)

#### Core Agent
| File | Role |
|------|------|
| `async_core.py` | Streaming agent with state machine, tool execution, reflection, learning |
| `core.py` | Synchronous agent (kept for compat) |
| `state.py` | AgentContext + AgentState dataclasses |
| `llm_client.py` | Sync LLM call wrapper |
| `async_llm_client.py` | Async streaming LLM call with tool-call delta accumulation |
| `hermes.py` | MCP Hermes agent (external tool server protocol) |

#### Security (14-vuln hardened)
| File | Role |
|------|------|
| `security.py` | API Key middleware with prod enforcement |
| `identity.py` | Identity RBAC + session tokens + rate limiting + fine-grained permissions |
| `tenancy.py` | Multi-tenant isolation with quota + API Key binding |
| `sandbox.py` | Process-level isolation with terminate/kill, path traversal prevention |
| `intrusion.py` | 5-type anomaly detection: brute force, path traversal, tenant hop, injection surge, tool anomaly |
| `governance.py` | AuditLogger (90d retention), PermissionManager (4-level), CircuitBreaker, CISOApprovalGate, SLOMonitor |
| `alerting.py` | Rule-based alerting with 12 rules including 4 security intrusion rules |

#### SuperAgent
| File | Role |
|------|------|
| `reflect_action.py` | Reflect→Action closed loop: auto-degrade failing tools after N failures, detect capability gaps |
| `debate.py` | Multi-model debate: Primary → Challenger → Rebuttal → Arbitrator → consensus |
| `bootstrap.py` | Skills bootstrapping: LLM generates tool code, compile() + AST safety check, auto-registers |

#### Orchestration
| File | Role |
|------|------|
| `orchestrator.py` | AgentOrchestrator with topological sort, voting, result aggregation |
| `crew_agent.py` | CrewAgent with identity/memory/tools/ReAct loop |
| `message_bus.py` | Agent-to-agent communication: direct, broadcast, delegate |
| `cross_reviewer.py` | Cross-model review: primary generates, reviewer checks, debate on disputes, human escalation |
| `attn_router.py` | Attention-based routing: residual/block/full modes, role-weighted signal aggregation |

#### Memory & Skills
| File | Role |
|------|------|
| `memory.py` | Persistent fact/lesson store with identity summarization |
| `auto_memory.py` | Automatic memory recording per operation |
| `context_compressor.py` | Conversation compression: truncate, summarize, hybrid |
| `skills.py` | AgentSkills standard (YAML frontmatter), AST-based tool discovery, self-creation |

#### Operations
| File | Role |
|------|------|
| `deploy.py` | Multi-environment deployment manager with versioning |
| `blue_green.py` | Blue-green zero-downtime deployment + smoke tests |
| `root_cause.py` | Root cause analyzer: trace-based failure pattern detection |
| `uptime.py` | Uptime tracking with MTTR |
| `cost_tracker.py` | Per-request cost estimation + daily/monthly budget caps |
| `reliability.py` | Reliability tracking with degradation detection |

#### Tools
| File | Role |
|------|------|
| `tools/registry.py` | ToolRegistry with Pydantic schema validation |
| `tools/calc_tool.py` | Calculator tool |
| `tools/web_search.py` | Web search tool (Bing) |
| `tools/file_ops.py` | File read/write operations |
| `tools/code_exec.py` | Python code execution |

### `ai_agent_playground/` — Framework layer (HuggingFace Transformers inspired)
| File | Role |
|------|------|
| `base.py` | BaseAgent with preprocess → _forward → postprocess pipeline |
| `config.py` | BaseAgentConfig dataclass |
| `llm.py` | LLMClient singleton |
| `constraints.py` | Constraint system for agent behavior |
| `evaluator_agent.py` | Evaluator for agent outputs |
| `human_loop.py` | Human-in-the-loop approval |
| `observability.py` | Observability primitives |
| `resilience.py` | Retry, fallback, timeout patterns |

### `observability/` — Observability stack
| File | Role |
|------|------|
| `tracer.py` | Distributed trace logging |
| `clear_metrics.py` | CLEAR panel: Cost, Latency, Efficacy, Assurance, Reliability |
| `monitoring.py` | Prometheus metrics export |

## Data Flow: A Complete Request

```
1. HTTP POST /chat/completions
       │
2. FastAPI middleware stack (reverse order):
   CORS → APIKey → Prometheus
       │
3. tenant_middleware: extract identity from Bearer token, bind tenant
       │  └── intrusion.record_tenant_access()
       │
4. check_prompt_injection() — all endpoints
       │  └── intrusion.record_injection_attempt()
       │
5. agent.run_stream(ctx, user_message)
       │
6. Agent loop (state machine):
   ┌─ PLANNING: call_llm_stream_async() → text or tool_calls
   │
   ├─ TOOL_CALL: sandbox.execute() wrapped by governance.wrap_tool_call()
   │   ├── identity.can() — fine-grained permission check
   │   ├── Path.resolve() — traversal prevention
   │   ├── multiprocessing.Process — timeout isolation with terminate/kill
   │   └── CircuitBreaker.before_call() — failure isolation
   │
   ├─ REFLECT: LLM self-critique → ReflectAction.evaluate()
   │   ├── Degrade tools with N+ consecutive failures
   │   └── Detect capability gaps → signal Bootstrap
   │
   └─ LEARN: Extract lesson → memory.add_lesson()
       └── Bootstrap.generate_from_reflection() if gap detected
           ├── LLM generates Python code
           ├── compile() + AST safety scan
           └── registry.register() → available next cycle
```

## Security Model

### Zero Trust Principles
1. **Explicit Verify**: API Key required in production, Bearer token on every request
2. **Least Privilege**: 4 roles + Resource-level PermissionGrant with fnmatch conditions
3. **Assume Breach**: Intrusion detection (5 types), audit trail (90d), rate limiting
4. **Defense in Depth**: Middleware → Identity → Sandbox → Governance → Intrusion Detection

### Risk Levels (tool → max runtime → policy)
| Level | Max Runtime | Write | Network | Requires Approval |
|-------|-------------|-------|---------|-------------------|
| low | 10s | no | no | no |
| medium | 20s | yes | yes | no |
| high | 30s | yes | yes | yes |
| critical | 60s | yes | yes | yes (CISO gate) |

## SuperAgent: Three Self-Improvement Loops

### 1. Reflect→Action
```
REFLECT text → ReflectActionEngine.evaluate()
  ├── Tool fails 3 times → degrade + substitute alternative
  ├── Same tool stuck in loop → pivot recommendation
  └── "I need X but don't have it" → signal Bootstrap
```

### 2. Multi-Model Debate
```
User Task
  → Primary model proposes solution
  → Challenger model critiques it
  → Primary rebuts critique
  → Arbitrator synthesizes consensus
```

### 3. Skills Bootstrapping
```
Capability gap detected
  → LLM generates tool code
  → compile() syntax check
  → AST safety scan (blocks os/subprocess/socket imports)
  → Write to skills/bootstrapped/
  → Registry.register() — available immediately
```

## Deployment

```bash
# Minimal (agent only)
docker-compose up -d

# Full (agent + Ollama + ChromaDB)
docker-compose --profile full up -d

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/super/status
```

## Key Design Decisions

1. **Pipeline pattern over framework**: BaseAgent with preprocess/forward/postprocess, inspired by HuggingFace Transformers — keeps the agent code readable and debuggable.

2. **Process isolation over thread isolation**: multiprocessing.Process with terminate()/kill() — threads can't be forcibly stopped in Python.

3. **Reflection drives action, not just logging**: Every REFLECT step feeds into ReflectActionEngine which modifies actual tool selection behavior.

4. **Cross-model over self-review**: Qwen reviewing DeepSeek catches blind spots that same-model review misses. Different architectures = different perspectives.

5. **Bootstrapping over pre-registration**: Agent discovers missing capabilities at runtime and generates them — this is Meta's HYPERAGENTS paper made concrete.

## Project File Count

```
agent/                43 Python files  (production engine)
ai_agent_playground/  23 Python files  (framework layer)
observability/         3 Python files  (monitoring)
scripts/              22 Python files  (ops/tools)
tests/                11 Python files  (133 test cases)
```
