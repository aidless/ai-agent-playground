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

## Autonomous Protocol (每次会话启动自动执行)

### 1. 启动即感知
- 读取 `memory/facts.json` 和 `memory/lessons.json`，加载项目记忆
- 运行 `git status --porcelain` 了解当前变更状态
- 检查最近的 3 条教训是否已应用到代码中

### 2. 主动推进
- 如果上次会话有未完成的任务，主动继续
- 发现项目问题（dead code、过时依赖、测试失败）不等用户问就修
- 小改进直接做，大改进说方案

### 3. 记忆回写
- 每次修复 bug 后写入 `memory/lessons.json`
- 每次发现新的项目约定后更新 CLAUDE.md 或 facts.json
- 会话结束前运行 `scripts/self_reflect.py` 总结 3 条关键教训

### 4. 联网兜底
- 不确定的技术问题用 `scripts/search_web.py` 搜索（Bing 通道）
- 优先用项目已有的工具，不重复造轮子

### 5. 工具调用路径
- Git Bash: `"C:/Program Files/Git/bin/bash.exe" -c "..."`
- Package manager: `uv run python ...`
- 搜索: `uv run python scripts/search_web.py search "..."`
