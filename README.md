# 🤖 AI Agent Playground / AI 智能体游乐场

**7 个 AI 智能体。一套管道模式。全部从零手写。**  
**7 AI agents. One Pipeline pattern. All built from scratch.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek_V4-green)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 📖 **这份 README 有中文和英文两个版本。往下翻就能看到中文。**  
> 📖 **This README is bilingual. Scroll down for your language.**

---

## 🎯 这是什么？（中文版）

一个 **AI 智能体作品集**。7 个能跑的项目，全部基于同一个 `BaseAgent` 管道模式。

> 管道模式是什么？就像餐厅服务员的工作流程：  
>   **接单**（把客人说的话写成小票）→ **做菜**（把票给厨房）→ **上菜**（端给客人）  
>   换成代码就是：**preprocess → _forward → postprocess**  
>   这个设计是从读 HuggingFace Transformers 源码学来的（5000+ 行）。

### 🚀 5 秒钟跑起来

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env        # 填上你的 DeepSeek 或 Anthropic API Key
uv sync                     # 自动装依赖
streamlit run app.py        # 打开浏览器 → http://localhost:8501
```

打开浏览器，你就能试用所有 7 个 Agent。不需要写一行代码。

### 🧠 7 个项目

| # | 项目 | 它做什么（像对奶奶解释一样） | 怎么跑 |
|---|------|--------------------------|--------|
| 1 | **Hello Agent** | 你问一句，AI 答一句。最简单的对话机器人。 | `uv run python -m hello_agent.agent` |
| 2 | **Code Review** | 把代码贴进去，AI 帮你看有没有 bug、安不安全、写得好不好。支持 15+ 种编程语言。 | `uv run python -m code_review_agent.main <目录>` |
| 3 | **RAG Q&A** | 上传 PDF/TXT 文件，然后对文件内容提问。AI 会先查文件再回答，答案带"出处"。就像学生翻课本答题。 | `uv run python -m rag_qa_system.main chat` |
| 4 | **Multi-Agent Crew** | 你说一句话需求（比如"做一个 todo 网站"），4 个 AI Agent 协作：产品经理拆任务 → 开发者写代码 → 测试查 bug → 运维出部署方案。 | `uv run python -m multi_agent_crew.main "..."` |
| 5 | **Resume Matcher** | 上传你的简历 + 粘贴职位描述 → AI 告诉你匹配度多少%、缺什么关键词、怎么改。求职神器。 | Streamlit → Resume Matcher 页 |
| 6 | **Mini-BERT** | 手写的一个 Transformer 模型（就是 ChatGPT 的"祖先"）。350 行代码，每行标注了张量形状。用来理解 AI 大脑怎么工作。 | `uv run python -m mini_bert.train` |
| 7 | **MCP Tool Agent** | 能用工具的 AI——自动搜索网页、读写文件、执行命令、算数学题。Claude Code 内部也是这么干的。 | `uv run python -m mcp_agent.main "..."` |

### 🏗️ 架构（像搭积木一样）

```
ai_agent_playground/         ← 核心框架（只 200 行，3 个文件）
├── config.py                ← 配置盒（就像设置面板："用哪个模型？说多少话？"）
├── base.py                  ← Agent 骨架（"接单→做菜→上菜"三步走模板）
└── llm.py                   ← 共享电话线（7 个 Agent 用同一根线打给 AI）

每个 Agent = config.py + agent.py + （可选的帮手文件）
7 个 Agent 共享同一个 BaseAgent 骨架，零重复代码。
```

### 🎓 从 Transformers 源码学的 5 个设计模式

| 模式 | 它是什么（奶奶版） | 在哪个文件 |
|------|------------------|-----------|
| **管道模板** | 服务员三步走：接单→做菜→上菜 | `base.py` |
| **配置驱动** | 手机参数表：屏幕 6.1寸、内存 256G——写清楚就行 | `config.py` |
| **分层组装** | 乐高积木：把小零件拼成大东西 | `code_review_agent/agent.py` |
| **编排分离** | 老板不干活，只安排"你先做A，然后他做B" | `multi_agent_crew/crew.py` |
| **共享单例** | 办公室只有一根电话线，大家排队用 | `llm.py` |

### 📂 项目结构

```
ai-agent-playground/
├── ai_agent_playground/     ← 核心框架（3 个文件，200 行）
├── hello_agent/             ← 项目 1：聊天
├── code_review_agent/       ← 项目 2：代码审查
├── rag_qa_system/           ← 项目 3：文档问答
├── multi_agent_crew/        ← 项目 4：虚拟开发团队
├── resume_matcher/          ← 项目 5：简历匹配
├── mini_bert/               ← 项目 6：手写 Transformer
├── mcp_agent/               ← 项目 7：工具使用 Agent
├── app.py                   ← 网页界面（Streamlit，浏览器打开就能用）
├── blog/                    ← 技术博客（中文 + 英文）
├── scripts/                 ← 自学习脚本
└── test_docs/               ← 测试用的文档
```

### 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 语言 | Python 3.11+ |
| 包管理 | uv（比 pip 快 10 倍） |
| AI 大脑 | DeepSeek V4 Pro（通过 Anthropic SDK 调用） |
| 向量数据库 | ChromaDB（存文档"意思"的地方，不是存文件） |
| 网页界面 | Streamlit（写 Python 就能出网页） |
| 深度学习 | PyTorch（手写 Mini-BERT 用的） |

### 👤 作者

**刘泽文** — 齐鲁理工学院 软件工程 2026 届

- GitHub: [@aidless](https://github.com/aidless)
- 博客: Dev.to · 掘金

> 学历不够，代码来凑。—— 我的求职信条

---

## 🎯 What This Is (English)

A portfolio of **7 production-style AI agents** — all sharing the same `BaseAgent` Pipeline pattern.

> Pipeline pattern explained: imagine a restaurant —  
>   **Take order** → **Cook** → **Serve**  
>   In code: **preprocess → _forward → postprocess**  
>   This pattern comes from reading 5000+ lines of HuggingFace Transformers source code.

### 🚀 Run in 5 Seconds

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env        # Add your DeepSeek or Anthropic API key
uv sync
streamlit run app.py        # Open http://localhost:8501
```

### 🧠 7 Projects (Grandma-Friendly Explanations)

| # | Project | What It Does (explain to your grandma) | Run |
|---|---------|--------------------------------------|-----|
| 1 | **Hello Agent** | You ask a question, AI answers. The simplest chatbot. | `uv run python -m hello_agent.agent` |
| 2 | **Code Review** | Paste code → AI finds bugs, security issues, and style problems. Supports 15+ languages. | `uv run python -m code_review_agent.main <path>` |
| 3 | **RAG Q&A** | Upload PDFs → ask questions → AI searches the docs first, then answers with citations. Like a student checking the textbook before answering. | `uv run python -m rag_qa_system.main chat` |
| 4 | **Multi-Agent Crew** | One sentence ("Build a todo app") → 4 AI agents collaborate: PM breaks it down → Dev writes code → QA reviews → DevOps deploys. | `uv run python -m multi_agent_crew.main "..."` |
| 5 | **Resume Matcher** | Upload resume + paste job description → AI tells you match %, missing keywords, and how to improve. Your personal career顾问. | Streamlit → Resume Matcher tab |
| 6 | **Mini-BERT** | A Transformer model written from scratch in 350 lines. Every tensor shape annotated. To understand how ChatGPT's "ancestor" works inside. | `uv run python -m mini_bert.train` |
| 7 | **MCP Tool Agent** | An AI that can USE tools — search the web, read/write files, run commands, calculate. This is how Claude Code works internally. | `uv run python -m mcp_agent.main "..."` |

### 🏗️ Architecture (Building Blocks)

```
ai_agent_playground/         ← Core framework (200 lines, 3 files)
├── config.py                ← Settings panel: "Which model? How many words?"
├── base.py                  ← Agent skeleton: "Take order → Cook → Serve"
└── llm.py                   ← Shared phone line: all 7 agents share one connection to AI

Each agent = config.py + agent.py + (optional helpers)
7 agents. 1 skeleton. Zero duplication.
```

### 🎓 5 Design Patterns (From Transformers Source Code)

| Pattern | Grandma Version | File |
|---------|----------------|------|
| **Pipeline** | Waiter's 3-step flow: take order → cook → serve | `base.py` |
| **Config-Driven** | Phone spec sheet: screen 6.1", storage 256GB — declare what, not how | `config.py` |
| **Layered Assembly** | LEGO bricks: snap small pieces into something big | `code_review_agent/agent.py` |
| **Orchestration** | Boss doesn't work — just says "you do A, then he does B" | `multi_agent_crew/crew.py` |
| **Shared Singleton** | One office phone line, everyone takes turns | `llm.py` |

### 👤 Author

**Liu Zewen (刘泽文)** — Software Engineering @ Qilu Institute of Technology (2026)

> "Credentials don't code. I do." — My job-hunting motto.

### 📄 License

MIT — use it, learn from it, build on it.
