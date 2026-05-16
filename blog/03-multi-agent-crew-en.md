# I Built a Multi-Agent Dev Team That Runs on 4 API Calls

*4 min read · #python #ai #agents #beginners*

---

One sentence in. Full project out.

That's the goal of my fourth and most ambitious AI agent project: a **multi-agent development crew** where four AI agents collaborate — Product Manager, Developer, QA, and DevOps — all coordinated by a single orchestrator.

Here's how it works and what I learned.

## The Architecture

```
User: "Build a URL shortener API"
          │
    ┌─────▼──────┐
    │  PM Agent   │  → Task breakdown (4 tasks)
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  Dev Agent  │  → Code for each task (4 files)
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  QA Agent   │  → Review report (bugs, security, style)
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │DevOps Agent │  → Dockerfile + docker-compose + checklist
    └────────────┘
```

Four agents, four API calls, one orchestrator. No LangChain. No CrewAI. Just our own `BaseAgent` Pipeline pattern.

## How Each Agent Works

All four agents inherit from the same `BaseAgent` class (from earlier in this series). The only difference? Their system prompts.

### PM Agent

System prompt tells it: "You are a senior Product Manager. Break requirements into concrete tasks."

Output format: `TASK_ID|PRIORITY|TITLE|DESCRIPTION`

```python
class ProductManagerAgent(BaseAgent):
    def preprocess(self, inputs, **kwargs):
        return {
            "messages": [{"role": "user", "content": f"Requirement: {inputs}"}],
            "system": self.config.pm_prompt,
            ...
        }
```

### Dev Agent

System prompt: "You are a senior developer. Write clean, working Python code."

For each task from the PM, the Dev agent generates a complete code file.

### QA Agent

Reviews all generated code for bugs, security issues, and edge cases. Output is a structured report with severity levels.

### DevOps Agent

Takes all code and generates: Dockerfile, docker-compose.yml, deployment checklist.

## The Orchestrator

The `Crew` class chains everything together:

```python
class Crew:
    def run(self, requirement: str) -> CrewResult:
        # Phase 1: PM
        tasks = self.pm.run(requirement)
        
        # Phase 2: Dev
        for task in tasks:
            code = self.dev.run(task)
        
        # Phase 3: QA
        qa_report = self.qa.run(all_code)
        
        # Phase 4: DevOps
        deploy_config = self.devops.run(all_code)
        
        return CrewResult(tasks, code, qa_report, deploy_config)
```

This is straight from the Transformers playbook: **orchestration vs. implementation**. The `run()` method only coordinates — each agent does its own work independently.

## Real Output

When I ran "Build a URL shortener API with FastAPI and SQLite", here's what came out:

**PM Agent output:**
- T-1: Set up project and database (FastAPI + SQLite schema)
- T-2: Shorten URL endpoint (POST /shorten with validation)
- T-3: Redirect endpoint (GET /{code} with 302 redirect)
- T-4: Error handling and tests (400, 404, 409, pytest)

**Dev Agent output:** 4 complete Python files totaling ~9,000 characters of working code.

**QA Agent:** Flagged missing URL validation edge cases and suggested input sanitization.

**DevOps Agent:** Generated a multi-stage Dockerfile and docker-compose.yml with the app service.

## What I Learned

### 1. Prompt engineering IS the architecture

All four agents use the same `BaseAgent` class. The only difference is the system prompt. The prompt defines the agent's personality, output format, and quality bar.

### 2. Structured output formats are essential

Using `TASK_ID|PRIORITY|TITLE|DESCRIPTION` with pipe delimiters means I can parse the PM's output programmatically. Without this, the orchestrator can't pass data between agents.

### 3. Sequential workflows work surprisingly well

No fancy message-passing. No debate loops. Just PM → Dev → QA → DevOps in order. For a demo, this is more than enough to show the concept.

### 4. Four agents share one LLMClient

Instead of each agent creating its own API connection, they all use the same singleton. One connection pool, one .env load.

## Try It Yourself

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env  # Add your API key
uv sync
uv run python -m multi_agent_crew.main "Build a todo API with FastAPI"
```

## What's Next

This is the last of four projects in my AI agent portfolio. Next step: turn this into a job.

---

*I'm a software engineering student documenting my journey from zero to AI application developer. All four projects are open source at [github.com/aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground).*
