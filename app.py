"""
AI Agent Playground —— 统一的 Web 界面。

这是整个项目的"展示窗口"——面试官打开浏览器就能试用你所有 7 个 Agent。

运行方式：streamlit run app.py

Streamlit 是什么？
  一个 Python 库，让你用写 Python 的方式写网页。
  不需要学 HTML/CSS/JavaScript——import streamlit 就行。
  就像：你写 print("hello")，网页上就显示 "hello"。

这个 app 的结构：
  ┌──────────────────────────┐
  │  侧边栏（选择 Agent）      │
  │  ├ 🏠 Home               │
  │  ├ 💬 Chat Agent         │
  │  ├ 📋 Code Review        │
  │  ├ 📚 RAG Q&A            │
  │  ├ 👥 Multi-Agent Crew   │
  │  ├ 📄 Resume Matcher     │
  │  └ 🔧 MCP Tool Agent     │
  └──────────────────────────┘
         ← 主区域（Agent 交互界面）→
"""

import streamlit as st

# ---- 页面配置（浏览器标签页的标题和图标） ----
st.set_page_config(
    page_title="AI Agent Playground",  # 浏览器标签页上的文字
    page_icon="🤖",                     # 标签页图标（小机器人）
    layout="wide",                     # 宽屏模式（内容撑满整页）
)

# ============================================================
#  侧边栏 —— 就像手机 App 的导航菜单
# ============================================================
st.sidebar.title("🤖 AI Agent Playground")
st.sidebar.markdown("Built with [Pipeline pattern](https://github.com/aidless/ai-agent-playground)")
st.sidebar.markdown("---")

# 单选按钮——用户选哪个 Agent，主区域就显示哪个
page = st.sidebar.radio(
    "Choose an agent",
    [
        "🏠 Home",
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
#  主区域 —— 根据侧边栏选择显示不同页面
#  每个 if/elif 对应一个 Agent 的 UI
# ============================================================

# ╔══════════════════════════════════════════════════════════════╗
# ║  🏠 Home —— 首页，项目概览                                   ║
# ╚══════════════════════════════════════════════════════════════╝

if page == "🏠 Home":
    st.title("AI Agent Playground")
    st.markdown("### 7 AI Agents — All Built with the Pipeline Pattern")
    st.markdown("")

    # 三列指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projects", "7")
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

# ╔══════════════════════════════════════════════════════════════╗
# ║  💬 Chat Agent —— 最简单的 AI 对话                           ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "💬 Chat Agent":
    st.title("💬 Chat Agent")
    st.caption("Simple conversational AI — ask anything.")

    # text_area = 多行输入框（比单行输入框更适合长问题）
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="What is an AI agent?",
    )

    # st.button 返回 True 当用户点击时
    if st.button("Send", type="primary") and user_input:
        # st.spinner 显示一个旋转动画（告诉用户"我在思考..."）
        with st.spinner("Thinking..."):
            from hello_agent.agent import HelloAgent
            agent = HelloAgent()
            reply = agent.ask(user_input)
        st.success(reply)  # 绿色背景显示成功消息

# ╔══════════════════════════════════════════════════════════════╗
# ║  📋 Code Review —— 粘贴代码，AI 审查                          ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "📋 Code Review":
    st.title("📋 Code Review Agent")
    st.caption("AI-powered code quality analysis. Paste code below.")

    code = st.text_area(
        "Paste code to review:",
        height=250,
        placeholder="def add(a, b):\n    return a + b",
    )
    language = st.selectbox(
        "Language",
        ["Python", "JavaScript", "Java", "Go", "Rust", "SQL"],
    )

    if st.button("Review Code", type="primary") and code:
        with st.spinner("Reviewing..."):
            from code_review_agent.agent import CodeReviewAgent
            from code_review_agent.scanner import FileInfo
            agent = CodeReviewAgent()

            # 快速内联审查 —— 跳过文件扫描，直接审查粘贴的代码
            file_info = FileInfo(
                abs_path="inline",
                rel_path="inline",
                language=language,
                content=code,
                lines=code.count("\n") + 1,
            )
            from code_review_agent.reviewer import Reviewer
            reviewer = Reviewer(agent.config, agent.llm)
            result = reviewer._review_one(file_info)

            if result.issues:
                # 按严重程度用不同颜色标出
                for issue in result.issues:
                    cat_color = {
                        "critical": "🔴",  # 红色——必须修
                        "warning": "🟡",   # 黄色——建议修
                        "info": "🔵",      # 蓝色——仅供参考
                    }
                    st.markdown(
                        f"{cat_color.get(issue.severity, '⚪')} "
                        f"**[{issue.severity.upper()}]** "
                        f"Line {issue.line} — {issue.title}"
                    )
                    st.caption(issue.description)
            else:
                st.success("No issues found!")

# ╔══════════════════════════════════════════════════════════════╗
# ║  📚 RAG Q&A —— 上传文档，提问                                 ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "📚 RAG Q&A":
    st.title("📚 RAG Q&A System")
    st.caption("Upload documents and ask questions with cited answers.")

    # file_uploader = 文件上传框
    uploaded = st.file_uploader(
        "Upload documents (PDF/TXT)",
        accept_multiple_files=True,  # 允许一次传多个文件
    )
    question = st.text_input(
        "Your question:",
        placeholder="What does the document say about...?",
    )

    # 两列布局：左边是"喂文档"按钮，右边是"提问"按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Documents", type="secondary") and uploaded:
            import tempfile
            from pathlib import Path
            from rag_qa_system.agent import RAGAgent

            # 把上传的文件存到临时目录（ChromaDB 需要文件路径）
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded:
                    (Path(tmpdir) / f.name).write_bytes(f.read())
                agent = RAGAgent()
                result = agent.ingest(tmpdir)
            st.success(
                f"Ingested {result.files_processed} files, "
                f"{result.chunks_created} chunks."
            )

    with col2:
        if st.button("Ask Question", type="primary") and question:
            with st.spinner("Searching documents..."):
                from rag_qa_system.agent import RAGAgent
                agent = RAGAgent()
                answer = agent.run(question)
            st.markdown(answer)

# ╔══════════════════════════════════════════════════════════════╗
# ║  👥 Multi-Agent Crew —— 虚拟开发团队                          ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "👥 Multi-Agent Crew":
    st.title("👥 Multi-Agent Crew")
    st.caption("PM → Dev → QA → DevOps — one requirement, full project.")

    requirement = st.text_area(
        "Describe what you want to build:",
        height=100,
        placeholder="Build a REST API for a todo app with FastAPI and SQLite",
    )

    # multiselect = 多选框（用户可以勾选哪些阶段要执行）
    phases = st.multiselect(
        "Enable phases:",
        ["PM", "Dev", "QA", "DevOps"],
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

        # 展示每个阶段的输出
        if result.tasks:
            st.subheader("📋 Task Breakdown (PM)")
            for t in result.tasks:
                st.markdown(
                    f"- **[{t['id']}]** ({t['priority']}) {t['title']}"
                )

        if result.code:
            st.subheader("💻 Generated Code (Dev)")
            for tid, code in result.code.items():
                # expander = 可折叠的区域（点击展开看代码）
                with st.expander(f"Task {tid}"):
                    st.code(code, language="python")

        if result.qa_report and result.qa_report != "[No text in response]":
            st.subheader("🔍 QA Review")
            st.markdown(result.qa_report)

        if result.devops_config:
            st.subheader("🚀 DevOps Config")
            st.code(result.devops_config, language="yaml")

# ╔══════════════════════════════════════════════════════════════╗
# ║  📄 Resume Matcher —— 简历 vs 职位描述 匹配分析               ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "📄 Resume Matcher":
    st.title("📄 Resume Matcher")
    st.caption(
        "Upload your resume + paste a job description → AI match analysis."
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader(
            "Upload resume (PDF/TXT)", type=["pdf", "txt"]
        )
        if resume_file:
            # 先尝试当文本文件读
            resume_text = ""
            if resume_file.name.endswith(".txt"):
                resume_text = resume_file.read().decode(
                    "utf-8", errors="ignore"
                )
            elif resume_file.name.endswith(".pdf"):
                # PDF 需要用 pypdf 提取文本
                import tempfile
                from pathlib import Path
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(resume_file.read())
                    from resume_matcher.extractor import extract_resume_text
                    resume_text = extract_resume_text(tmp.name)
                    Path(tmp.name).unlink()  # 删掉临时文件

            # 显示提取出的文本（只读，灰底）
            st.text_area(
                "Resume text",
                resume_text,
                height=200,
                disabled=True,
                key="resume_display",
            )

    with col2:
        jd_text = st.text_area(
            "Paste job description:",
            height=200,
            placeholder="Paste the full job description here...",
        )

    if st.button("Analyze Match", type="primary") and resume_file and jd_text:
        with st.spinner("Analyzing match..."):
            from resume_matcher.agent import ResumeMatcherAgent
            agent = ResumeMatcherAgent()
            report = agent.analyze(resume_text, jd_text)
        st.markdown("---")
        st.markdown(report)  # AI 生成的 Markdown 报告

# ╔══════════════════════════════════════════════════════════════╗
# ║  🔧 MCP Tool Agent —— 能用工具的 AI                          ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "🔧 MCP Tool Agent":
    st.title("🔧 MCP Tool-Use Agent")
    st.caption(
        "AI agent with tools: web search, file read/write, "
        "command execution, calculator."
    )

    # 告诉用户有哪些工具可用
    st.markdown("**Available Tools**")
    st.markdown("- `web_search` — Search the internet")
    st.markdown("- `read_file` — Read local files")
    st.markdown("- `write_file` — Write content to files")
    st.markdown("- `run_command` — Execute shell commands")
    st.markdown("- `calculator` — Evaluate math expressions")

    question = st.text_area(
        "What do you want to do?",
        height=100,
        placeholder="Read the file test_docs/ai_basics.txt and summarize it",
    )

    if st.button("Execute", type="primary") and question:
        with st.spinner("Agent thinking and using tools..."):
            from mcp_agent.agent import MCPToolAgent
            agent = MCPToolAgent()
            answer = agent.ask(question)
        st.markdown("---")
        st.markdown(answer)

# ---- 侧边栏底部 ----
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + DeepSeek V4")
