# ai-agent-playground — CLAUDE.md

## Project purpose
AI agent portfolio for job hunting. 4 projects demonstrating AI application development skills.
Built with Pipeline pattern inspired by HuggingFace Transformers source code.

## Tech stack
- Python 3.11+, uv package manager
- DeepSeek V4 Pro via Anthropic SDK (base_url: https://api.deepseek.com/anthropic)
- ChromaDB for RAG vector storage (all-MiniLM-L6-v2 embeddings)
- pypdf for PDF loading
- Environment: Windows 11, Git Bash

## Architecture (Transformers-inspired)
```
ai_agent_playground/     ← Core framework (config + base + llm — 200 lines)
  config.py              ← BaseAgentConfig dataclass (like PreTrainedConfig)
  base.py                ← BaseAgent with preprocess→_forward→postprocess (like Pipeline)
  llm.py                 ← LLMClient singleton (like shared PreTrainedModel)
```

Every agent: inherit BaseAgent, implement 3 pipeline methods.

## Projects
| # | Package | Run command |
|---|---------|-------------|
| 1 | hello_agent | `uv run python -m hello_agent.agent` |
| 2 | code_review_agent | `uv run python -m code_review_agent.main <path>` |
| 3 | rag_qa_system | `uv run python -m rag_qa_system.main chat` |
| 4 | multi_agent_crew | `uv run python -m multi_agent_crew.main "requirement"` |

## Key files to know
- `.env` — API keys (gitignored), template at `.env.example`
- `reports/` — generated review reports (gitignored)
- `chroma_db/` — vector database (gitignored)
- `blog/` — technical blog posts (EN + ZH)

## Conventions
- Always use `uv run python -m <module>` to run, not `python <file>`
- Always format with `import guards`: try relative import except absolute import
- Configs are dataclasses, not dicts or module-level vars
- One agent = one package with config.py + agent.py + (optional helpers)

## Network notes
- GitHub/PyPI sometimes blocked in China — use Tsinghua mirror:
  `uv add <pkg> --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- GitHub Desktop is installed for pushing when CLI fails
