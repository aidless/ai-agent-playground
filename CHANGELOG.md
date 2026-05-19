# Changelog

All notable changes to AI Agent Playground.

## [1.3.0] — 2026-05-19

### Added
- OpenTelemetry integration (`observability/telemetry.py`) with OTLP export
- A2A Protocol (`agent/a2a.py`) — Google-style agent-to-agent communication
- Cross-platform utilities (`agent/platform_utils.py`)
- Memory poison detection (`agent/poison_detector.py`) — 2-layer defense
- Goal drift detector (`agent/goal_tracker.py`) — keyword overlap tracking
- Cost tracker session reporting with real-time budget warnings
- 20-round self-play training with competence curves
- Hard benchmark suite (code repair, b3 security, multi-agent, pipeline)
- Engine benchmark (baseline vs debate vs matrix comparison)
- Security penetration test (14/14 scenarios)
- b3 Security benchmark (10/10 WDTA-style attacks)
- Self-correction feedback loop (30% improvement on failed fixes)
- Comprehensive deployment guide (DEPLOY.md)
- deploy.sh — one-command deploy script
- Technical blog posts (CN + EN)

### Changed
- ARCHITECTURE.md with 8-layer stack diagram
- README with Mermaid diagram, badges, API table
- Code repair benchmark: LLM-based validation (0% → 70% → 90% fix rate)
- b3 security: 80% → 90% → 100% attack blocking
- Cost tracker enhanced with session-level tracking

### Security
- 14/14 penetration test scenarios pass
- 10/10 b3 benchmark attacks blocked
- 30+ prompt injection patterns (CN + EN)
- 5-type intrusion detection
- 12-pattern secret scanner
- Resource-level fine-grained permissions
- API key production enforcement

## [1.2.0] — 2026-05-18

### Added
- SuperAgent engines: debate, evolution, bootstrap, reflect-action
- AutoPilot autonomous coordinator (9-engine orchestration)
- Agent Matrix — multi-model specialized routing
- Unified Pipeline — Crew → Debate → CrossReview
- Self-Play training with Generator learning + Memory consolidation
- Evaluation Gate — 3D quality scoring (Interface + Functional + Utility)
- Sandboxed Meta Self-Evolution
- Sandbox process isolation (multiprocessing.Process + terminate/kill)
- 5-type intrusion detection system
- Secret scanner (12 regex patterns)
- Fine-grained RBAC with Resource-level permissions
- API key ↔ tenant binding
- 161 tests (zero failures)

### Security
- 14 vulnerabilities fixed (OWASP Top 10 for LLM Apps)
- Path traversal prevention (normPath + case-insensitive)
- Token signature upgraded to 256-bit HMAC-SHA256
- Rate limiting (5/min/IP) on token validation
- Prompt injection detection on all endpoints
- Input length limits (100KB max)
- CORS restricted by environment
- Middleware execution order fixed
- Tool risk levels corrected
- Audit log redaction (API keys, JWTs, Bearer tokens)

## [1.1.0] — 2026-05-10

### Added
- Multi-model debate engine (competition mode)
- Cross-model reviewer (DeepSeek + Qwen2.5)
- Crew orchestration with topological sort
- AttnRes routing (residual/block/full modes)
- Blue-green deployment
- Root cause analyzer
- CLEAR 5D monitoring panel
- Uptime/SLO tracking
- 66 tests passing

## [1.0.0] — 2026-04-15

### Added
- Pipeline-pattern agent (HuggingFace Transformers inspired)
- AsyncAgent with streaming (SSE)
- Tool Registry with Pydantic validation
- State machine: IDLE → PLANNING → TOOL_CALL → REFLECT → LEARN → DONE
- 4 project demo: Hello Agent, Code Review, RAG Q&A, Multi-Agent Crew
- Governance panel: audit, permissions, circuit breaker
- Multi-tenant isolation
- Identity RBAC
- 56 tests passing
