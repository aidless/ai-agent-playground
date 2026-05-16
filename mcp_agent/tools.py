"""Tool implementations for the MCP agent.

Each tool is a standalone function — the agent can call them independently.
This is the core of agentic AI: the LLM decides WHAT to call, the agent executes HOW.
"""

import json
import math
import os
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


def web_search(query: str, num_results: int = 3) -> str:
    """Search the web and return results.

    Uses DuckDuckGo's HTML endpoint (no API key needed).
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "MCP-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

        # Simple extraction of result snippets
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
        return f"Search failed: {e}. Try again later."


def read_file(path: str) -> str:
    """Read and return file contents."""
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


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"Write failed: {e}"


def run_command(command: str, timeout: int = 30) -> str:
    """Run a shell command and return stdout+stderr."""
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


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/().%^ eEπ ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"

    safe_expr = expression.replace("^", "**").replace("π", str(math.pi))
    try:
        result = eval(safe_expr, {"__builtins__": {}}, {
            "math": math, "sqrt": math.sqrt, "sin": math.sin,
            "cos": math.cos, "log": math.log, "pi": math.pi,
            "e": math.e, "pow": math.pow, "abs": abs, "round": round,
        })
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# Tool registry — all available tools
TOOLS = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "calculator": calculator,
}

TOOL_DESCRIPTIONS = {
    "web_search": "Search the web. Args: query (str), num_results (int, optional)",
    "read_file": "Read a file. Args: path (str)",
    "write_file": "Write to a file. Args: path (str), content (str)",
    "run_command": "Run a shell command. Args: command (str), timeout (int, optional)",
    "calculator": "Evaluate math. Args: expression (str)",
}
