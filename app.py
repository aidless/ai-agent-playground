"""
AI Agent Playground —— 统一的 Web 界面。

12 个 Agent + Harness Engineering 系统仪表盘。
Built with the Pipeline pattern from HuggingFace Transformers.

Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Agent Playground",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
#  Sidebar
# ============================================================
st.sidebar.title("🤖 AI Agent Playground")
st.sidebar.markdown("Built with [Pipeline pattern](https://github.com/aidless/ai-agent-playground)")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a module",
    [
        "🏠 Home",
        "⚙️ Harness Dashboard",
        "💬 Chat Agent",
        "📋 Code Review",
        "📚 RAG Q&A",
        "👥 Multi-Agent Crew",
        "📄 Resume Matcher",
        "🔧 MCP Tool Agent",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("By Liu Zewen | [GitHub](https://github.com/aidless)")


# ============================================================
#  Home
# ============================================================

if page == "🏠 Home":
    st.title("AI Agent Playground")
    st.markdown("### 12 Agents · 6 Harness Modules · 1 Production System")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Agents", "12")
    with col2:
        st.metric("Harness Modules", "6")
    with col3:
        st.metric("Code Size", "~8000 LOC")
    with col4:
        st.metric("Tech Stack", "Python + DeepSeek")

    st.markdown("---")
    st.markdown("""
    ### Architecture
    ```
    ┌──────── Harness Engineering Layer ────────┐
    │  State Manager  │  Human Loop  │  Eval Gate │
    │  Observability  │  Evaluator   │ Constraints │
    └────────────────────────────────────────────┘
    ┌──────── Agent Runtime Layer ──────────────┐
    │  MCP Protocol  │  Docker Sandbox  │  Stream  │
    └────────────────────────────────────────────┘
    ┌──────── Knowledge Layer ──────────────────┐
    │  RAG  │  Hybrid Search  │  Re-ranker  │  Chunking  │
    └────────────────────────────────────────────┘
    ┌──────── Foundation ───────────────────────┐
    │  preprocess() → _forward() → postprocess() │
    │  Inspired by HuggingFace Transformers      │
    └────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")
    st.markdown("### Quick Links")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("[GitHub](https://github.com/aidless/ai-agent-playground)")
    with cols[1]:
        st.markdown("[Tutorial](https://github.com/aidless/ai-agent-playground/blob/master/TUTORIAL.md)")
    with cols[2]:
        st.markdown("[Reports](https://github.com/aidless/ai-agent-playground/tree/master/reports)")


# ============================================================
#  Harness Dashboard
# ============================================================

elif page == "⚙️ Harness Dashboard":
    st.title("⚙️ Harness Engineering Dashboard")
    st.caption("Real-time system observability, constraints, and gate status.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Observability", "🔒 Safety", "📋 State", "✅ Eval Gate"
    ])

    with tab1:
        st.subheader("LLM Observability — Trace & Metrics")
        try:
            from ai_agent_playground.observability import get_tracer
            tracer = get_tracer()
            snap = tracer.snapshot()

            cols = st.columns(4)
            with cols[0]:
                st.metric("Traces", snap.total_traces)
            with cols[1]:
                err_rate = (snap.error_count / snap.total_traces * 100) if snap.total_traces else 0
                st.metric("Error Rate", f"{err_rate:.1f}%")
            with cols[2]:
                st.metric("Avg Latency", f"{snap.avg_latency_ms:.0f}ms")
            with cols[3]:
                st.metric("LLM Tokens", snap.total_llm_tokens)

            st.markdown("---")
            cols = st.columns(3)
            with cols[0]:
                st.metric("P50 Latency", f"{snap.p50_latency_ms:.0f}ms")
            with cols[1]:
                st.metric("P95 Latency", f"{snap.p95_latency_ms:.0f}ms")
            with cols[2]:
                st.metric("P99 Latency", f"{snap.p99_latency_ms:.0f}ms")

            st.caption("Run `uv run python -m demo.production_demo` to generate traces.")
        except Exception as e:
            st.info(f"Observability not yet initialized. Run a demo to start. ({e})")

    with tab2:
        st.subheader("Human-in-the-Loop — Approval Gate")
        try:
            from ai_agent_playground.human_loop import ApprovalGate, Policy

            if "approval_gate" not in st.session_state:
                st.session_state.approval_gate = ApprovalGate()
                st.session_state.approval_gate.set_policies({
                    "read_file": Policy.AUTO_APPROVE,
                    "calculator": Policy.AUTO_APPROVE,
                    "write_file": Policy.ALWAYS_ASK,
                    "run_command": Policy.ALWAYS_ASK,
                })

            gate = st.session_state.approval_gate
            report = gate.report()

            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Checks", report["total_approvals"])
            with cols[1]:
                st.metric("Approval Rate", f"{report['approval_rate']:.0%}")
            with cols[2]:
                st.metric("Denied", report["denied"])

            st.markdown("**Policy Configuration**")
            for tool, policy in gate._policies.items():
                st.markdown(f"- `{tool}` → {policy.value}")

            st.caption("Policies configurable at `ai_agent_playground/human_loop.py`")
        except Exception as e:
            st.info(f"Approval gate info. ({e})")

    with tab3:
        st.subheader("State Manager — Task Checklist")
        try:
            from ai_agent_playground.state_manager import StateManager
            import tempfile

            if "state_mgr" not in st.session_state:
                tmp = tempfile.mkdtemp(prefix="agent_state_")
                sm = StateManager(work_dir=tmp)
                sm.start_session("Demo session — view Harness state")
                sm.add_tasks([
                    ("init", "Initialize agent workspace"),
                    ("verify", "Verify tool availability"),
                    ("execute", "Execute primary task"),
                    ("validate", "Run post-execution validation"),
                    ("report", "Generate completion report"),
                ])
                sm.start_task("init")
                sm.complete_task("init", "Workspace initialized successfully")
                sm.start_task("verify")
                sm.complete_task("verify", "All tools available: calculator, read_file, sandbox")
                st.session_state.state_mgr = sm

            sm = st.session_state.state_mgr
            for item in sm.manifest.items:
                icon = {"completed": "✅", "in_progress": "🔄", "failed": "❌",
                        "pending": "⬜", "skipped": "⏭️"}[item.status]
                st.markdown(f"{icon} **[{item.id}]** {item.description}")
                if item.result:
                    st.caption(f"  → {item.result[:200]}")

            st.metric("Progress", f"{sm.manifest.progress_pct:.0%}")
            st.caption(f"Session: {sm.manifest.session_id}")
        except Exception as e:
            st.info(f"State manager info. ({e})")

    with tab4:
        st.subheader("Eval Gate — Quality Baseline")
        try:
            import json
            from pathlib import Path

            baseline_path = Path("reports/baseline.json")
            if baseline_path.exists():
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                st.markdown("| Agent | Score | Pass Rate | Cases |")
                st.markdown("|-------|-------|-----------|-------|")
                for agent, scores in baseline.items():
                    st.markdown(
                        f"| {agent} | {scores['avg_score']:.3f} | "
                        f"{scores['pass_rate']:.0%} | {scores['total_cases']} |"
                    )
            else:
                st.info("No baseline yet. Run `uv run python scripts/eval_gate.py`")
        except Exception as e:
            st.info(f"Eval gate info. ({e})")


# ============================================================
#  Chat Agent
# ============================================================

elif page == "💬 Chat Agent":
    st.title("💬 Chat Agent")
    st.caption("Simple conversational AI — ask anything.")

    user_input = st.text_area(
        "Your message:", height=100,
        placeholder="What is an AI agent?",
    )

    if st.button("Send", type="primary") and user_input:
        with st.spinner("Thinking..."):
            from hello_agent.agent import HelloAgent
            agent = HelloAgent()
            reply = agent.ask(user_input)
        st.success(reply)


# ============================================================
#  Code Review
# ============================================================

elif page == "📋 Code Review":
    st.title("📋 Code Review Agent")
    st.caption("AI-powered code quality analysis. Paste code below.")

    code = st.text_area(
        "Paste code to review:", height=250,
        placeholder="def add(a, b):\n    return a + b",
    )
    language = st.selectbox(
        "Language", ["Python", "JavaScript", "Java", "Go", "Rust", "SQL"],
    )

    if st.button("Review Code", type="primary") and code:
        with st.spinner("Reviewing..."):
            from code_review_agent.agent import CodeReviewAgent
            from code_review_agent.scanner import FileInfo
            agent = CodeReviewAgent()
            file_info = FileInfo(
                abs_path="inline", rel_path="inline",
                language=language, content=code,
                lines=code.count("\n") + 1,
            )
            from code_review_agent.reviewer import Reviewer
            reviewer = Reviewer(agent.config, agent.llm)
            result = reviewer._review_one(file_info)
            if result.issues:
                for issue in result.issues:
                    cat_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                    st.markdown(
                        f"{cat_color.get(issue.severity, '⚪')} "
                        f"**[{issue.severity.upper()}]** "
                        f"Line {issue.line} — {issue.title}"
                    )
                    st.caption(issue.description)
            else:
                st.success("No issues found!")


# ============================================================
#  RAG Q&A
# ============================================================

elif page == "📚 RAG Q&A":
    st.title("📚 RAG Q&A System")
    st.caption("Upload documents and ask questions with cited answers.")

    from rag_qa_system.config import RAGConfig
    config = RAGConfig()

    st.markdown(f"**Config**: `{config.search_mode}` search, "
                f"`{config.chunk_strategy}` chunking, "
                f"top_k={config.top_k}")

    uploaded = st.file_uploader(
        "Upload documents (PDF/TXT)", accept_multiple_files=True,
    )
    question = st.text_input(
        "Your question:",
        placeholder="What does the document say about...?",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Documents", type="secondary") and uploaded:
            import tempfile
            from pathlib import Path
            from rag_qa_system.agent import RAGAgent
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded:
                    (Path(tmpdir) / f.name).write_bytes(f.read())
                agent = RAGAgent()
                result = agent.ingest(tmpdir)
            st.success(
                f"Ingested {result.files_processed} files, "
                f"{result.chunks_created} chunks "
                f"({result.strategy})."
            )
    with col2:
        if st.button("Ask Question", type="primary") and question:
            with st.spinner("Searching documents..."):
                from rag_qa_system.agent import RAGAgent
                agent = RAGAgent()
                answer = agent.run(question)
            st.markdown(answer)


# ============================================================
#  Multi-Agent Crew
# ============================================================

elif page == "👥 Multi-Agent Crew":
    st.title("👥 Multi-Agent Crew")
    st.caption("PM → Dev → QA → DevOps — one requirement, full project.")

    requirement = st.text_area(
        "Describe what you want to build:", height=100,
        placeholder="Build a REST API for a todo app with FastAPI and SQLite",
    )
    phases = st.multiselect(
        "Enable phases:", ["PM", "Dev", "QA", "DevOps"],
        default=["PM", "Dev", "QA", "DevOps"],
    )

    if st.button("Run Crew", type="primary") and requirement:
        from multi_agent_crew.config import CrewConfig
        from multi_agent_crew.crew import Crew
        config = CrewConfig(
            enable_qa="QA" in phases,
            enable_devops="DevOps" in phases,
        )
        crew = Crew(config)
        with st.spinner("PM breaking down tasks..."):
            result = crew.run(requirement)
        if result.tasks:
            st.subheader("📋 Task Breakdown (PM)")
            for t in result.tasks:
                st.markdown(f"- **[{t['id']}]** ({t['priority']}) {t['title']}")
        if result.code:
            st.subheader("💻 Generated Code (Dev)")
            for tid, code in result.code.items():
                with st.expander(f"Task {tid}"):
                    st.code(code, language="python")
        if result.qa_report and result.qa_report != "[No text in response]":
            st.subheader("🔍 QA Review")
            st.markdown(result.qa_report)
        if result.devops_config:
            st.subheader("🚀 DevOps Config")
            st.code(result.devops_config, language="yaml")


# ============================================================
#  Resume Matcher
# ============================================================

elif page == "📄 Resume Matcher":
    st.title("📄 Resume Matcher")
    st.caption("Upload resume + paste job description → AI match analysis.")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload resume (PDF/TXT)", type=["pdf", "txt"])
        resume_text = ""
        if resume_file:
            if resume_file.name.endswith(".txt"):
                resume_text = resume_file.read().decode("utf-8", errors="ignore")
            elif resume_file.name.endswith(".pdf"):
                import tempfile
                from pathlib import Path
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resume_file.read())
                    from resume_matcher.extractor import extract_resume_text
                    resume_text = extract_resume_text(tmp.name)
                    Path(tmp.name).unlink()
            st.text_area("Resume text", resume_text, height=200,
                         disabled=True, key="resume_display")
    with col2:
        jd_text = st.text_area(
            "Paste job description:", height=200,
            placeholder="Paste the full job description here...",
        )

    if st.button("Analyze Match", type="primary") and resume_file and jd_text:
        with st.spinner("Analyzing match..."):
            from resume_matcher.agent import ResumeMatcherAgent
            agent = ResumeMatcherAgent()
            report = agent.analyze(resume_text, jd_text)
        st.markdown("---")
        st.markdown(report)


# ============================================================
#  MCP Tool Agent (upgraded: streaming + sandbox + MCP)
# ============================================================

elif page == "🔧 MCP Tool Agent":
    st.title("🔧 MCP Tool-Use Agent")
    st.caption("AI agent with MCP protocol, Docker sandbox, and streaming output.")

    tab1, tab2 = st.tabs(["Agent", "Config & Status"])

    with tab1:
        st.markdown("**Available Tools**")
        cols = st.columns(6)
        with cols[0]:
            st.markdown("🌐 `web_search`")
        with cols[1]:
            st.markdown("📖 `read_file`")
        with cols[2]:
            st.markdown("✏️ `write_file`")
        with cols[3]:
            st.markdown("⚡ `run_command`")
        with cols[4]:
            st.markdown("🔢 `calculator`")
        with cols[5]:
            st.markdown("🐳 `sandbox_execute`")

        use_streaming = st.toggle("Streaming output", value=True,
                                  help="Show tokens as they're generated")
        use_mcp = st.toggle("MCP mode", value=False,
                            help="Connect to external MCP server for dynamic tool discovery")

        question = st.text_area(
            "What do you want to do?", height=100,
            placeholder="Calculate 15*15+12, then write result to calc_result.txt",
        )

        if st.button("Execute", type="primary") and question:
            from mcp_agent.config import MCPAgentConfig

            mcp_cmd = None
            if use_mcp:
                mcp_cmd = ["uv", "run", "python", "-m", "mcp_agent.mcp_server"]

            config = MCPAgentConfig(mcp_command=mcp_cmd)

            if use_streaming:
                from mcp_agent.agent import MCPToolAgent
                from ai_agent_playground.base import ToolCallEvent

                agent_obj = MCPToolAgent(config)
                output_placeholder = st.empty()
                full_output = ""

                try:
                    with st.spinner("Agent thinking..."):
                        for item in agent_obj.run_stream(question):
                            if isinstance(item, str):
                                full_output += item
                                output_placeholder.markdown(full_output)
                            elif isinstance(item, ToolCallEvent):
                                if item.phase == "start":
                                    args_str = ", ".join(
                                        f"{k}={v}" for k, v in (item.args or {}).items()
                                    )
                                    full_output += f"\n\n⚙️ `{item.tool_name}({args_str})` → "
                                else:
                                    preview = (item.result or "")[:150].replace("\n", " ")
                                    full_output += f"*{preview}*\n\n"
                                output_placeholder.markdown(full_output)
                finally:
                    agent_obj.close()
            else:
                from mcp_agent.agent import MCPToolAgent
                agent_obj = MCPToolAgent(config)
                try:
                    with st.spinner("Agent thinking and using tools..."):
                        answer = agent_obj.ask(question)
                    st.markdown("---")
                    st.markdown(answer)
                finally:
                    agent_obj.close()

    with tab2:
        st.markdown("**System Status**")
        try:
            from mcp_agent.sandbox import DockerSandbox
            docker_ok = DockerSandbox.is_available()
        except Exception:
            docker_ok = False
        st.metric("Docker Sandbox", "Available" if docker_ok else "Not available")

        try:
            from mcp_agent.mcp_client import MCPClient
            st.metric("MCP Protocol", "Available (stdlib implementation)")
        except Exception:
            st.metric("MCP Protocol", "Not loaded")

        st.markdown("---")
        st.caption("Configuration: `mcp_agent/config.py`")


# ---- Sidebar footer ----
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + DeepSeek V4")
st.sidebar.caption("Liu Zewen 2026")
