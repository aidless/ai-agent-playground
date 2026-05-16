# 🤖 AI Agent Playground

**5 AI agents. One Pipeline pattern. All built from scratch.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek_V4-green)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-5%2F5%20Projects%20Complete-brightgreen)]()

---

## 🎯 What This Is

A portfolio of **production-style AI agents** built with a shared `BaseAgent` Pipeline pattern — inspired by reading 5000+ lines of HuggingFace Transformers source code.

> **Every agent**: `preprocess → _forward → postprocess`  
> **Every config**: typed dataclass, one per agent  
> **One LLM client**: shared across all agents  

## 🚀 Quick Demo

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env        # Add your DeepSeek or Anthropic API key
uv sync
streamlit run app.py        # Open http://localhost:8501
```

## 🧠 Projects

| # | Agent | What It Does | Run |
|---|-------|-------------|-----|
| 1 | **Hello Agent** | Conversational AI with system prompts & multi-turn memory | `uv run python -m hello_agent.agent` |
| 2 | **Code Review** | Scans 15+ languages, flags bugs/security/style issues via AI | `uv run python -m code_review_agent.main <path>` |
| 3 | **RAG Q&A** | PDF → chunk → embed → ChromaDB → cited answers | `uv run python -m rag_qa_system.main chat` |
| 4 | **Multi-Agent Crew** | PM → Dev → QA → DevOps. One sentence → full project | `uv run python -m multi_agent_crew.main "..."` |
| 5 | **Resume Matcher** | Resume vs JD analysis: match %, missing keywords, suggestions | `streamlit run app.py` → Resume Matcher tab |

## 🏗️ Architecture

```
ai_agent_playground/         ← Core framework (200 lines, 3 files)
├── config.py                ← BaseAgentConfig (like PreTrainedConfig)
├── base.py                  ← BaseAgent with Pipeline pattern (like Pipeline)
└── llm.py                   ← Shared LLMClient singleton

Each agent = config.py + agent.py + (optional helpers)
All sharing the same BaseAgent, 0 duplication.
```

## 🎓 Design Patterns (from Transformers Source Code)

| Pattern | Source | Implementation |
|---------|--------|---------------|
| Pipeline Template | `pipelines/base.py` | `BaseAgent.run()` → preprocess → _forward → postprocess |
| Config-Driven | `configuration_utils.py` | `BaseAgentConfig` dataclass with typed defaults |
| Layered Assembly | `modeling_bert.py` | Scanner → Reviewer → Reporter as pluggable components |
| Orchestration | `generation/utils.py` | Agent orchestrates; components implement |
| Shared Singleton | `modeling_utils.py` | `LLMClient` one instance, all agents share |

[→ Read the full breakdown (blog)](blog/)

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Package Manager | uv |
| LLM | DeepSeek V4 Pro (Anthropic SDK compatible) |
| Vector DB | ChromaDB (all-MiniLM-L6-v2 embeddings) |
| UI | Streamlit |
| PDF | pypdf |
| Infra | Git, GitHub, Windows 11 |

## 📂 Project Structure

```
ai-agent-playground/
├── ai_agent_playground/     ← Core framework
├── hello_agent/             ← Project 1: Chat
├── code_review_agent/       ← Project 2: Code Review
├── rag_qa_system/           ← Project 3: RAG Q&A
├── multi_agent_crew/        ← Project 4: Multi-Agent
├── resume_matcher/          ← Project 5: Resume Matcher
├── app.py                   ← Streamlit web UI (all 5 agents)
├── blog/                    ← Technical blog posts (EN + ZH)
├── scripts/                 ← Self-learning analysis tools
├── test_docs/               ← Sample documents for testing
└── reports/                 ← Generated outputs (gitignored)
```

## 📝 Blog Posts

| # | Topic | EN | ZH |
|---|-------|----|----|
| 1 | I Built a Code Review Agent in 2 Hours | [EN](blog/01-code-review-agent-en.md) | [中文](blog/01-code-review-agent-zh.md) |
| 2 | 5 Design Patterns from Transformers Source | [EN](blog/02-transformers-patterns-en.md) | [中文](blog/02-transformers-patterns-zh.md) |
| 3 | Multi-Agent Dev Team on 4 API Calls | [EN](blog/03-multi-agent-crew-en.md) | [中文](blog/03-multi-agent-crew-zh.md) |

## 👤 Author

**Liu Zewen (刘泽文)** — Software Engineering @ Qilu Institute of Technology (2026)

- GitHub: [@aidless](https://github.com/aidless)
- Blog: Dev.to · 掘金

Building real projects to prove that code speaks louder than credentials.

## 📄 License

MIT — use it, learn from it, build on it.
