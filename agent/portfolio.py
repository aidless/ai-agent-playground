"""Unified Portfolio — 4 AI Agent projects showcase for job hunting.

Serves a professional portfolio page with:
  - Live metrics from the running agent system
  - 4 project cards with tech stack and features
  - Enterprise readiness score
  - Contact / resume section
"""

import json
import os
from datetime import datetime


PROJECTS = [
    {
        "id": "hello_agent",
        "name": "Hello Agent",
        "tagline": "Pipeline-pattern AI agent with streaming, tools, and state machine",
        "emoji": "",
        "description": "Base agent implementing the HuggingFace Transformers-inspired pipeline pattern. Features async streaming, tool registry, state machine (IDLE→PLANNING→TOOL_CALL→DONE), and full observability with trace_id.",
        "tech": ["Python 3.11+", "FastAPI", "AsyncIO", "DeepSeek API", "SSE Streaming", "Uvicorn"],
        "highlights": [
            "Pipeline pattern: preprocess → _forward → postprocess",
            "Async streaming with SSE (Server-Sent Events)",
            "Tool Registry with Pydantic parameter validation",
            "State machine: IDLE → PLANNING → TOOL_CALL → REFLECT → LEARN → DONE",
        ],
        "run": "uv run python -m hello_agent.agent",
        "status": "stable",
    },
    {
        "id": "code_review_agent",
        "name": "Code Review Agent",
        "tagline": "AI-powered multi-dimensional code quality analysis",
        "emoji": "",
        "description": "Automated code review agent that scans source code across 5 dimensions: security, performance, style, testing, and architecture. Generates structured review reports with actionable suggestions.",
        "tech": ["Python", "AST Parser", "DeepSeek API", "Markdown Reports", "CLI"],
        "highlights": [
            "5-dimensional code analysis: Security, Performance, Style, Testing, Architecture",
            "AST-based static analysis without executing code",
            "Structured Markdown report generation",
            "CLI tool: uv run python -m code_review_agent.main <path>",
        ],
        "run": "uv run python -m code_review_agent.main <file_or_dir>",
        "status": "stable",
    },
    {
        "id": "rag_qa_system",
        "name": "RAG Q&A System",
        "tagline": "Document QA with hybrid search, reranking, and cited answers",
        "emoji": "",
        "description": "Retrieval-Augmented Generation system for document question answering. Load PDFs, chunk text, build vector index with ChromaDB, and answer questions with citations using hybrid search (BM25 + semantic) and cross-encoder reranking.",
        "tech": ["Python", "ChromaDB", "all-MiniLM-L6-v2", "pypdf", "BM25", "Cross-encoder"],
        "highlights": [
            "Hybrid search: BM25 (keyword) + ChromaDB (semantic) fusion",
            "Cross-encoder reranking for retrieval precision",
            "Cited answers with source document references",
            "PDF ingestion pipeline with smart chunking",
            "Interactive chat mode",
        ],
        "run": "uv run python -m rag_qa_system.main chat",
        "status": "beta",
    },
    {
        "id": "multi_agent_crew",
        "name": "Multi-Agent Crew",
        "tagline": "PM → Dev → QA → DevOps collaborative AI team",
        "emoji": "",
        "description": "Multi-agent collaboration system where specialized AI agents (Project Manager, Developer, QA, DevOps) work together on software tasks. Uses a message bus for agent communication, crew orchestration, and voting-based result aggregation.",
        "tech": ["Python", "AsyncIO", "Message Bus", "CrewAI Pattern", "DeepSeek API"],
        "highlights": [
            "4 specialized agent roles: PM, Developer, QA, DevOps",
            "Message bus: direct, broadcast, and delegate communication",
            "Crew orchestration with task decomposition and result aggregation",
            "Voting-based consensus for final answers",
            "Each agent has independent memory and state",
        ],
        "run": 'uv run python -m multi_agent_crew.main "your requirement"',
        "status": "beta",
    },
]


ENTERPRISE_FEATURES = [
    {"name": "Sandbox Execution", "icon": "S", "desc": "4-tier risk isolation"},
    {"name": "CISO Approval Gate", "icon": "C", "desc": "Formal security review"},
    {"name": "Multi-Tenant", "icon": "M", "desc": "Quota + namespace isolation"},
    {"name": "Gray Release", "icon": "G", "desc": "Canary with auto-rollback"},
    {"name": "SLOMonitor", "icon": "S", "desc": "Error budget tracking"},
    {"name": "Circuit Breaker", "icon": "CB", "desc": "Auto-failure isolation"},
    {"name": "Identity RBAC", "icon": "I", "desc": "4-role permission model"},
    {"name": "Prometheus", "icon": "P", "desc": "Full observability stack"},
    {"name": "CI/CD Pipeline", "icon": "CI", "desc": "GitHub Actions"},
    {"name": "71 Tests", "icon": "T", "desc": "100% pass rate"},
]


def portfolio_html() -> str:
    stars = ""

    # Project cards
    project_cards = ""
    for i, p in enumerate(PROJECTS):
        tag_class = "stable" if p["status"] == "stable" else "beta"
        highlights = "".join(f'<li>{h}</li>' for h in p["highlights"])
        tech_tags = "".join(f'<span class="tech-tag">{t}</span>' for t in p["tech"])
        project_cards += f"""
        <div class="project-card">
            <div class="project-header">
                <span class="project-emoji">{p['emoji']}</span>
                <div>
                    <h3>{p['name']}</h3>
                    <p class="tagline">{p['tagline']}</p>
                </div>
                <span class="status-tag {tag_class}">{p['status']}</span>
            </div>
            <p class="desc">{p['description']}</p>
            <div class="tech-stack">{tech_tags}</div>
            <ul class="highlights">{highlights}</ul>
            <code class="run-cmd">{p['run']}</code>
        </div>"""

    # Enterprise feature badges
    feat_badges = "".join(
        f'<div class="feat-badge"><span class="feat-icon">{f["icon"]}</span><span class="feat-name">{f["name"]}</span><span class="feat-desc">{f["desc"]}</span></div>'
        for f in ENTERPRISE_FEATURES
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Liu Zewen — AI Agent Portfolio</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0e14; color:#c9d1d9; font:14px/1.6 -apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif; }}
.container {{ max-width:1200px; margin:0 auto; padding:32px 24px; }}

/* Hero */
.hero {{ text-align:center; padding:48px 0 32px; }}
.hero h1 {{ font-size:36px; color:#e6edf3; margin-bottom:8px; }}
.hero .sub {{ color:#8b949e; font-size:16px; margin-bottom:16px; }}
.hero .badges {{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }}
.badge {{ padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; }}
.badge.green {{ background:#23863622; color:#56d364; border:1px solid #238636; }}
.badge.blue {{ background:#1f6feb22; color:#79c0ff; border:1px solid #1f6feb; }}
.badge.purple {{ background:#bc8cff22; color:#bc8cff; border:1px solid #8957e5; }}
.badge.orange {{ background:#d2992222; color:#d29922; border:1px solid #d29922; }}

/* Metrics row */
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:24px 0 32px; }}
.metric {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; text-align:center; }}
.metric .val {{ font-size:28px; font-weight:bold; }}
.metric .lbl {{ color:#8b949e; font-size:11px; margin-top:4px; }}
.metric .val.green {{ color:#56d364; }}
.metric .val.blue {{ color:#79c0ff; }}
.metric .val.orange {{ color:#d29922; }}

/* Section titles */
.section-title {{ font-size:20px; color:#e6edf3; margin:40px 0 16px; padding-bottom:8px; border-bottom:1px solid #21262d; }}

/* Project cards */
.projects {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:16px; }}
.project-card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:24px; transition:border-color .2s; }}
.project-card:hover {{ border-color:#58a6ff; }}
.project-header {{ display:flex; align-items:flex-start; gap:12px; margin-bottom:12px; }}
.project-emoji {{ font-size:28px; }}
.project-header h3 {{ font-size:18px; color:#58a6ff; }}
.tagline {{ color:#8b949e; font-size:12px; margin-top:2px; }}
.status-tag {{ padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; margin-left:auto; flex-shrink:0; }}
.status-tag.stable {{ background:#23863622; color:#56d364; border:1px solid #238636; }}
.status-tag.beta {{ background:#d2992222; color:#d29922; border:1px solid #d29922; }}
.desc {{ color:#8b949e; font-size:13px; margin-bottom:12px; }}
.tech-stack {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }}
.tech-tag {{ background:#1f6feb11; color:#79c0ff; border:1px solid #1f6feb44; padding:2px 8px; border-radius:4px; font-size:11px; }}
.highlights {{ color:#c9d1d9; font-size:12px; padding-left:18px; margin-bottom:12px; }}
.highlights li {{ margin-bottom:3px; }}
.run-cmd {{ display:block; background:#0d1117; color:#7ee787; padding:8px 12px; border-radius:6px; font-size:11px; border:1px solid #21262d; }}

/* Enterprise features */
.enterprise {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:8px; margin-top:16px; }}
.feat-badge {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; display:flex; align-items:center; gap:8px; }}
.feat-icon {{ width:28px; height:28px; background:#1f6feb22; color:#58a6ff; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:bold; flex-shrink:0; }}
.feat-name {{ font-weight:600; font-size:13px; color:#e6edf3; }}
.feat-desc {{ color:#8b949e; font-size:11px; }}

/* Footer */
.footer {{ text-align:center; padding:40px 0 24px; color:#484f58; font-size:11px; }}
.footer a {{ color:#58a6ff; text-decoration:none; }}
.contact {{ display:flex; gap:16px; justify-content:center; margin:16px 0; }}
.contact-item {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 20px; font-size:13px; }}
.contact-item span {{ color:#8b949e; }} .contact-item strong {{ color:#e6edf3; }}

/* Responsive */
@media(max-width:1100px){{ .projects{{grid-template-columns:1fr;}} }}
@media(max-width:600px){{ .hero h1{{font-size:24px;}} .metrics{{grid-template-columns:repeat(2,1fr);}} }}
</style>
</head>
<body>
<div class="container">

<!-- Hero -->
<div class="hero">
    <h1>Liu Zewen 刘泽文</h1>
    <p class="sub">AI Application Developer | AI Agent Engineer | 2026 Graduate</p>
    <div class="badges">
        <span class="badge green">71 Tests Passing</span>
        <span class="badge blue">Python + FastAPI + DeepSeek</span>
        <span class="badge purple">Enterprise P1 78%</span>
        <span class="badge orange">Qilu Institute of Technology</span>
    </div>
</div>

<!-- Live Metrics -->
<div class="metrics" id="metrics">
    <div class="metric"><div class="val green" id="m-status">-</div><div class="lbl">System Status</div></div>
    <div class="metric"><div class="val blue" id="m-uptime">-</div><div class="lbl">Uptime</div></div>
    <div class="metric"><div class="val green" id="m-tests">71/71</div><div class="lbl">Tests Passing</div></div>
    <div class="metric"><div class="val orange" id="m-cost">$0.02</div><div class="lbl">Cost Today</div></div>
    <div class="metric"><div class="val blue" id="m-efficacy">90%</div><div class="lbl">Efficacy</div></div>
    <div class="metric"><div class="val green" id="m-subsystems">8/8</div><div class="lbl">Subsystems Healthy</div></div>
</div>

<!-- 4 Projects -->
<h2 class="section-title">Projects</h2>
<div class="projects">
{project_cards}
</div>

<!-- Enterprise Features -->
<h2 class="section-title">Enterprise Readiness</h2>
<div class="enterprise">{feat_badges}</div>

<!-- Contact -->
<div class="contact">
    <div class="contact-item"><span>GitHub</span> <strong>aidless</strong></div>
    <div class="contact-item"><span>Location</span> <strong>Shandong, China</strong></div>
    <div class="contact-item"><span>School</span> <strong>Qilu Institute of Technology</strong></div>
    <div class="contact-item"><span>Degree</span> <strong>B.Eng. Software Engineering 2026</strong></div>
</div>

<div class="footer">
    Built with Python, FastAPI, DeepSeek API, ChromaDB &mdash; AI Agent Playground &copy; 2026
</div>

</div>

<script>
// Fetch live metrics from server
async function loadMetrics() {{
    try {{
        let r = await fetch('/health');
        let h = await r.json();
        document.getElementById('m-status').textContent = h.status;
        document.getElementById('m-subsystems').textContent = h.healthy;

        r = await fetch('/uptime');
        let u = await r.json();
        document.getElementById('m-uptime').textContent = u.uptime_pct.toFixed(1) + '%';

        r = await fetch('/clear');
        let c = await r.json();
        document.getElementById('m-cost').textContent = '$' + c.cost.today_usd.toFixed(2);
        document.getElementById('m-efficacy').textContent = (c.efficacy.success_rate * 100).toFixed(0) + '%';
    }} catch(e) {{}}
}}
loadMetrics();
setInterval(loadMetrics, 10000);
</script>
</body>
</html>"""
