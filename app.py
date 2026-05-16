"""AI Agent Playground — Streamlit web app.

Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Agent Playground",
    page_icon="🤖",
    layout="wide",
)

# ---- Sidebar ----
st.sidebar.title("🤖 AI Agent Playground")
st.sidebar.markdown("Built with [Pipeline pattern](https://github.com/aidless/ai-agent-playground)")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose an agent",
    ["🏠 Home", "💬 Chat Agent", "📋 Code Review", "📚 RAG Q&A", "👥 Multi-Agent Crew", "📄 Resume Matcher"],
)

st.sidebar.markdown("---")
st.sidebar.caption("By Liu Zewen | [GitHub](https://github.com/aidless)")

# ---- Pages ----

if page == "🏠 Home":
    st.title("AI Agent Playground")
    st.markdown("### 4 AI Agents + Resume Matcher — All Built with the Pipeline Pattern")
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projects", "5")
        st.metric("Design Patterns", "5")
    with col2:
        st.metric("Core Framework", "200 LOC")
        st.metric("GitHub Stars", "⭐ Coming soon")
    with col3:
        st.metric("Tech Stack", "Python + DeepSeek")
        st.metric("Deployed", "Streamlit Cloud")

    st.markdown("---")
    st.markdown("""
    ### Architecture
    ```
    User Input → preprocess() → _forward() → postprocess() → Result
                  (prepare)      (AI call)      (format)
    ```
    Inspired by [HuggingFace Transformers](https://github.com/huggingface/transformers) source code.
    """)

elif page == "💬 Chat Agent":
    st.title("💬 Chat Agent")
    st.caption("Simple conversational AI — ask anything.")

    user_input = st.text_area("Your message:", height=100, placeholder="What is an AI agent?")
    if st.button("Send", type="primary") and user_input:
        with st.spinner("Thinking..."):
            from hello_agent.agent import HelloAgent
            agent = HelloAgent()
            reply = agent.ask(user_input)
        st.success(reply)

elif page == "📋 Code Review":
    st.title("📋 Code Review Agent")
    st.caption("AI-powered code quality analysis. Paste code below.")

    code = st.text_area("Paste code to review:", height=250,
                        placeholder="def add(a, b):\n    return a + b")
    language = st.selectbox("Language", ["Python", "JavaScript", "Java", "Go", "Rust", "SQL"])

    if st.button("Review Code", type="primary") and code:
        with st.spinner("Reviewing..."):
            from code_review_agent.agent import CodeReviewAgent
            from code_review_agent.scanner import FileInfo
            agent = CodeReviewAgent()

            # Quick inline review — bypass scanner
            file_info = FileInfo(
                abs_path="inline", rel_path="inline",
                language=language, content=code, lines=code.count("\n") + 1,
            )
            from code_review_agent.reviewer import Reviewer
            reviewer = Reviewer(agent.config, agent.llm)
            result = reviewer._review_one(file_info)
            if result.issues:
                for issue in result.issues:
                    cat_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                    st.markdown(f"{cat_color.get(issue.severity, '⚪')} **[{issue.severity.upper()}]** "
                                f"Line {issue.line} — {issue.title}")
                    st.caption(issue.description)
            else:
                st.success("No issues found!")

elif page == "📚 RAG Q&A":
    st.title("📚 RAG Q&A System")
    st.caption("Upload documents and ask questions with cited answers.")

    uploaded = st.file_uploader("Upload documents (PDF/TXT)", accept_multiple_files=True)
    question = st.text_input("Your question:", placeholder="What does the document say about...?")

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
            st.success(f"Ingested {result.files_processed} files, {result.chunks_created} chunks.")

    with col2:
        if st.button("Ask Question", type="primary") and question:
            with st.spinner("Searching documents..."):
                from rag_qa_system.agent import RAGAgent
                agent = RAGAgent()
                answer = agent.run(question)
            st.markdown(answer)

elif page == "👥 Multi-Agent Crew":
    st.title("👥 Multi-Agent Crew")
    st.caption("PM → Dev → QA → DevOps — one requirement, full project.")

    requirement = st.text_area(
        "Describe what you want to build:",
        height=100,
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

elif page == "📄 Resume Matcher":
    st.title("📄 Resume Matcher")
    st.caption("Upload your resume + paste a job description → AI match analysis.")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload resume (PDF/TXT)", type=["pdf", "txt"])
        if resume_file:
            resume_text = resume_file.read().decode("utf-8", errors="ignore") if resume_file.name.endswith(".txt") else ""
            if resume_file.name.endswith(".pdf"):
                import tempfile
                from pathlib import Path
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resume_file.read())
                    from resume_matcher.extractor import extract_resume_text
                    resume_text = extract_resume_text(tmp.name)
                    Path(tmp.name).unlink()
            st.text_area("Resume text", resume_text, height=200, disabled=True, key="resume_display")
    with col2:
        jd_text = st.text_area("Paste job description:", height=200,
                               placeholder="Paste the full job description here...")

    if st.button("Analyze Match", type="primary") and resume_file and jd_text:
        with st.spinner("Analyzing match..."):
            from resume_matcher.agent import ResumeMatcherAgent
            agent = ResumeMatcherAgent()
            report = agent.analyze(resume_text, jd_text)
        st.markdown("---")
        st.markdown(report)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit + DeepSeek V4")
