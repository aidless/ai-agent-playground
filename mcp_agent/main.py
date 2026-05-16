"""MCP Tool-Use Agent CLI.

Usage:
  uv run python -m mcp_agent.main "What is the latest Python version?"
  uv run python -m mcp_agent.main "Calculate sqrt(144) + 3^3"
  uv run python -m mcp_agent.main "Read the file test_docs/ai_basics.txt and summarize it"
  uv run python -m mcp_agent.main                    (demo mode)
"""

import sys
from .agent import MCPToolAgent


def main():
    print("=" * 60)
    print("  MCP Tool-Use Agent")
    print("  Tools: web_search, read_file, write_file, run_command, calculator")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Search the web for 'Python 3.13 new features' and tell me the top 3"

    print(f"Q: {question}\n")
    agent = MCPToolAgent()
    answer = agent.ask(question)
    # Print safely (avoid GBK encoding errors on Windows)
    try:
        print(answer)
    except UnicodeEncodeError:
        print(answer.encode('ascii', errors='replace').decode('ascii'))


if __name__ == "__main__":
    main()
