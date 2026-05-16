"""ai-agent-playground — Building AI agents, one project at a time.

Built with the Pipeline pattern (inspired by HuggingFace Transformers):
  preprocess → _forward → postprocess

Projects:
  1. hello_agent       — First contact with the Claude API
  2. code_review_agent — AI-powered code review
  3. rag_qa_system     — Retrieval-augmented Q&A (coming soon)
  4. multi_agent_crew  — Multi-agent collaboration (coming soon)

Run a project:
  uv run python -m hello_agent.agent
  uv run python -m code_review_agent.main
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
