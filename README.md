# ai-agent-playground

Building AI agents with Python, LangChain, and Claude API — one project at a time.

> **Author**: Liu Zewen
> **Goal**: Land an AI application developer role by building real, deployable AI agent projects.

## Projects

| # | Project | Status | Description |
|---|---------|--------|-------------|
| 1 | `hello_agent/` | Done | First contact with the API — chat, system prompts, multi-turn conversation |
| 2 | `code_review_agent/` | Planned | AI agent that reviews GitHub repos and generates code quality reports |
| 3 | `rag_qa_system/` | Planned | Upload PDFs → vector search → citation-backed Q&A |
| 4 | `multi_agent_crew/` | Planned | Multi-agent collaboration with CrewAI — PM → Dev → QA → DevOps |

## Tech Stack

- **Language**: Python 3.11+
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **LLM**: DeepSeek (via Anthropic-compatible API)
- **Agent Frameworks**: LangChain, CrewAI (planned)
- **Vector DB**: ChromaDB
- **UI**: Streamlit, Gradio
- **Infra**: GitHub Actions, Docker (planned)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/lzw-ai/ai-agent-playground.git
cd ai-agent-playground

# 2. Install dependencies
uv sync

# 3. Set up API key
cp .env.example .env
# Edit .env and add your API key

# 4. Run the first agent
uv run python hello_agent/main.py
```

## Learning Log

I'm documenting my AI engineering journey on:
- [Dev.to](https://dev.to/) — English technical blogs
- [掘金](https://juejin.cn/) — Chinese technical blogs

Follow along if you're on a similar path.

## License

MIT
