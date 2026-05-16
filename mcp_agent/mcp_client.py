"""
Lightweight MCP Client — talks to any MCP server via JSON-RPC over stdio.

Handles the MCP handshake (initialize), tool discovery (tools/list),
and tool invocation (tools/call). No external MCP SDK dependency —
just Python stdlib + JSON-RPC 2.0.

Usage:
    with MCPClient(["uv", "run", "python", "-m", "mcp_agent.mcp_server"]) as client:
        tools = client.list_tools()
        result = client.call_tool("web_search", {"query": "Python 3.13"})
"""

import json
import subprocess
from typing import Any


class MCPError(RuntimeError):
    """Raised when the MCP server returns an error."""


class MCPClient:
    """Lightweight MCP JSON-RPC client that drives a subprocess MCP server."""

    def __init__(self, command: list[str]):
        self.command = command
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._server_info: dict = {}
        self._started = False

    # ----- Lifecycle -----

    def start(self):
        if self._started:
            return
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._handshake()
        self._started = True

    def close(self):
        if not self._process:
            return
        try:
            self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()

    # ----- JSON-RPC internals -----

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _write(self, data: dict):
        line = json.dumps(data, ensure_ascii=False)
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> str:
        while True:
            line = self._process.stdout.readline()
            if not line:
                stderr_output = ""
                try:
                    stderr_output = self._process.stderr.read()
                except Exception:
                    pass
                raise MCPError(
                    f"MCP server process exited unexpectedly. stderr: {stderr_output[:500]}"
                )
            line = line.strip()
            if line:
                return line

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        self._write(req)
        return self._read_response(req["id"])

    def _send_notification(self, method: str, params: dict | None = None):
        notif = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._write(notif)

    def _read_response(self, request_id: int) -> dict:
        """Read lines until we get the response matching our request ID.

        Skips notifications (messages without an id field).
        """
        while True:
            line = self._read_line()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "id" not in msg:
                continue

            if msg["id"] == request_id:
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(
                        f"MCP error [{err.get('code', '?')}]: {err.get('message', str(err))}"
                    )
                return msg.get("result", {})

    # ----- MCP protocol -----

    def _handshake(self):
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-agent", "version": "0.1.0"},
            },
        )
        self._server_info = result
        self._send_notification("notifications/initialized")

    # ----- Public API -----

    def list_tools(self) -> list[dict]:
        """Discover available tools.

        Returns a list of dicts, each containing:
          name, description, inputSchema
        """
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool on the MCP server.

        Returns the text output produced by the tool.
        """
        result = self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        parts = []
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else json.dumps(result)
