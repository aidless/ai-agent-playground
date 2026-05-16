# I Built an AI Code Review Agent in 2 Hours — Here's How

*Published on [Dev.to](https://dev.to/) · 5 min read*

---

I'm a software engineering student about to graduate. Like many juniors, I was worried about the job market. So instead of waiting, I decided to build things — real AI projects that solve real problems.

The first project: an **AI-powered code review agent** that scans your codebase and flags bugs, security issues, and style problems. All in under 200 lines of Python.

Here's how I built it, what I learned, and why you should try it too.

## The Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.11 |
| AI Model | DeepSeek V4 Pro (via Anthropic SDK) |
| Package Manager | uv |
| Output | Markdown report |

Why DeepSeek instead of Claude directly? It's cheaper and the Anthropic-compatible endpoint makes migration trivial. The same code works with Claude by changing one line.

## Architecture: 3 Simple Modules

```
code_review_agent/
├── scanner.py    # Walk directories, find code files
├── reviewer.py   # Send each file to the AI
├── report.py     # Generate Markdown report
└── main.py       # Wire everything together
```

No fancy frameworks. No LangChain. Just three functions chained together. I believe in starting simple and adding complexity only when needed.

### 1. Scanner — Finding the Code

The scanner walks a directory tree and collects every file the AI can review:

```python
CODE_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".java": "Java", ".go": "Go", ".rs": "Rust", ".sql": "SQL",
    # ... 10+ more
}

def scan_directory(root: str) -> list[FileInfo]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in CODE_EXTENSIONS:
                files.append(FileInfo(...))
    return files
```

Key decisions:
- **Skip dirs like .git, node_modules, .venv** — no need to review third-party code
- **200KB size limit** — don't burn API credits on generated files
- **Sort by language then path** — makes the report predictable

### 2. Reviewer — The AI Brain

This is where the magic happens. Each file gets sent to the AI with a carefully designed system prompt:

```python
REVIEW_SYSTEM_PROMPT = """\
You are a senior code reviewer. Analyze the code below and report issues.

For each issue you find, use EXACTLY this format:
  SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION

SEVERITY: critical, warning, info
CATEGORY: bug, security, performance, style, best-practice
LINE: approximate line number
"""
```

The pipe-delimited format is intentional. It's easy to parse, easy for the AI to follow, and human-readable at a glance.

**What I learned about prompt design:**
1. **Be specific about output format** — "use EXACTLY this format" prevents rambling
2. **Give concrete examples** — the AI follows patterns better than instructions
3. **"Only report real issues"** — without this, the AI invents problems to seem helpful

### 3. Report — Making It Readable

The report module takes parsed issues and generates clean Markdown. Issues are grouped by severity: critical first, then warnings, then suggestions. Each issue links back to the file and line number.

## Running It

```bash
# Review any codebase
uv run python -m code_review_agent.main /path/to/project

# Review itself (demo mode)
uv run python -m code_review_agent.main
```

## The First Run — and a Surprise

I ran it on its own source code. It found 4 issues in `pyproject.toml`:

- "anthropic>=0.102.0 doesn't exist on PyPI" — **Wrong**. Version 0.102.0 exists.
- "python-dotenv>=1.2.2 doesn't exist" — **Also wrong**. I literally installed it two hours ago.
- "Missing langchain dependency" — Technically true, but I intentionally removed it.

**Lesson: AI code review is a second pair of eyes, not a judge.** It catches things you miss, but it also hallucinates. Always verify.

## What's Next

This is project 2 of 4 in my [ai-agent-playground](https://github.com/aidless/ai-agent-playground) series:

1. **Hello Agent** — First contact with the API
2. **Code Review Agent** — You just read about it
3. **RAG Q&A System** — Upload PDFs, ask questions with citations (coming soon)
4. **Multi-Agent Crew** — PM → Developer → QA → DevOps collaboration

## Try It Yourself

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env  # Add your API key
uv sync
uv run python -m code_review_agent.main /path/to/your/project
```

---

*I'm documenting my journey from software engineering student to AI application developer. If you're on a similar path, let's connect — I'll be posting weekly updates here and on [GitHub](https://github.com/aidless).*
