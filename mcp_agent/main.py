"""MCP Tool-Use Agent CLI.

Usage:
  # Local mode (built-in tools)
  uv run python -m mcp_agent.main "Calculate sqrt(144) + 3^3"

  # MCP mode (connect to an MCP server)
  uv run python -m mcp_agent.main --mcp "uv run python -m mcp_agent.mcp_server" "Search the web for Python 3.13 features"

  # Demo mode
  uv run python -m mcp_agent.main
"""

import sys

from .agent import MCPToolAgent
from .config import MCPAgentConfig


def main():
    print("=" * 60)
    print("  MCP Tool-Use Agent")
    print("=" * 60)
    print()

    args = sys.argv[1:]
    mcp_command = None
    question = None

    if "--mcp" in args:
        idx = args.index("--mcp")
        mcp_args = []
        for a in args[idx + 1:]:
            if a.startswith("--"):
                break
            mcp_args.append(a)
        mcp_command = mcp_args
        remaining = []
        skip = False
        for a in args:
            if a == "--mcp":
                skip = True
                continue
            if skip:
                if a.startswith("--"):
                    remaining.append(a)
                    skip = False
                else:
                    continue
            else:
                remaining.append(a)
        args = remaining

    if args:
        question = " ".join(args)
    else:
        question = "Search the web for 'Python 3.13 new features' and tell me the top 3"

    if mcp_command:
        print(f"MCP Server: {' '.join(mcp_command)}")
    else:
        print("Mode: local (built-in tools)")
    print(f"Q: {question}\n")

    config = MCPAgentConfig(mcp_command=mcp_command)
    agent = MCPToolAgent(config)
    try:
        answer = agent.ask(question)
        try:
            print(answer)
        except UnicodeEncodeError:
            print(answer.encode("ascii", errors="replace").decode("ascii"))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
