"""
Official MCP Server — exposes your AI tools via the Model Context Protocol.

This is a PROPER MCP server (not just our own tool-use agent).
Any MCP client (Claude Code, Cursor, VS Code, custom apps) can connect and use:

  Tools exposed:
    - web_search:   Search the web via DuckDuckGo
    - read_file:    Read local files
    - write_file:   Write to local files
    - run_command:  Execute shell commands
    - calculator:   Safe math evaluation
    - sandbox_execute: Safely run Python code in isolated Docker container

Run as:
  uv run python -m mcp_agent.mcp_server

Or register in Claude Code:
  claude mcp add my-tools -s user -- uv run python -m mcp_agent.mcp_server

Built with the official MCP Python SDK (mcp >= 1.0).
"""

import json
import math
import os
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .sandbox import get_sandbox, SandboxError


# ---- Tool Implementations ----

def _web_search(query: str, num_results: int = 3) -> str:
    """DuckDuckGo search — no API key needed."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "MCP-Tools/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

        results = []
        for line in html.split("\n"):
            if 'class="result__snippet"' in line:
                snippet = line.split('class="result__snippet">')[1].split("<")[0].strip()
                results.append(snippet)
            if len(results) >= num_results:
                break

        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))

    except Exception as e:
        return f"Search failed: {e}"


def _read_file(path: str) -> str:
    """Read a file. Returns content (truncated at 5000 chars)."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"File not found: {path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > 5000:
            content = content[:5000] + f"\n\n... [truncated, {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Read failed: {e}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"Write failed: {e}"


def _run_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(Path.home()),
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n[stderr]\n" + result.stderr.strip()
        return output[:3000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Command failed: {e}"


def _calculator(expression: str) -> str:
    """Safely evaluate math."""
    allowed = set("0123456789+-*/().%^ eEπ ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters. Use only: digits, +, -, *, /, (, ), ., ^"
    safe = expression.replace("^", "**").replace("π", str(math.pi))
    try:
        result = eval(safe, {"__builtins__": {}}, {
            "math": math, "sqrt": math.sqrt, "sin": math.sin,
            "cos": math.cos, "log": math.log, "pi": math.pi,
            "e": math.e, "pow": math.pow, "abs": abs, "round": round,
        })
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def _sandbox_execute(code: str, language: str = "python") -> str:
    """Execute Python code in a secure Docker sandbox.

    The container has no network access, limited memory (256MB),
    limited CPU (1 core), and a read-only filesystem.
    """
    try:
        s = get_sandbox()
        return s.execute(code, language)
    except SandboxError as e:
        return f"Sandbox error: {e}"


# ---- MCP Server Setup ----

server = Server("ai-agent-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns top results as text snippets. No API key required.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific for better results."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (default: 3, max: 10)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_file",
            description="Read a file from the local filesystem. Returns file contents as text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file."
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path where to write the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    },
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="run_command",
            description="Execute a shell command and return stdout/stderr. Timeout: 30 seconds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="calculator",
            description="Safely evaluate a mathematical expression. Supports: +, -, *, /, (), ^, sqrt, sin, cos, log, pi, e.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. 'sqrt(144) + 3^3'"
                    },
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="sandbox_execute",
            description="Safely execute Python code in an isolated Docker container. The container has no network access, 256MB memory limit, 1 CPU core, read-only filesystem, and a 30-second timeout. Use this to run generated code safely without risking the host system.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python source code to execute in the sandbox."
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (currently only 'python' is supported)",
                        "default": "python",
                    },
                },
                "required": ["code"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to the right implementation."""
    tool_map = {
        "web_search": lambda: _web_search(**arguments),
        "read_file": lambda: _read_file(**arguments),
        "write_file": lambda: _write_file(**arguments),
        "run_command": lambda: _run_command(**arguments),
        "calculator": lambda: _calculator(**arguments),
        "sandbox_execute": lambda: _sandbox_execute(**arguments),
    }

    if name not in tool_map:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        result = tool_map[name]()
        return [TextContent(type="text", text=str(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Tool error: {e}")]


# ---- Entry Point ----

async def main():
    """Run the MCP server over stdio (Claude Code connects via stdin/stdout)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
