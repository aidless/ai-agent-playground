"""A2A Protocol — Agent-to-Agent communication standard.

Based on Google A2A protocol. Enables interoperable communication between
agents via JSON-RPC 2.0 over HTTP. Each agent exposes its capabilities
as an 'agent card' that other agents can discover.

Supports:
  - Agent card discovery
  - Task sending with JSON-RPC 2.0
  - Capability matching
  - Security validation before task acceptance
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

A2A_DIR = Path(__file__).resolve().parent.parent / "memory" / "a2a"


@dataclass
class AgentCard:
    """Public capability declaration for an agent."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    endpoints: dict[str, str] = field(default_factory=dict)
    security_level: str = "standard"  # standard | high | critical
    owner: str = ""
    contact: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "endpoints": self.endpoints,
            "security_level": self.security_level,
            "owner": self.owner,
            "contact": self.contact,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            endpoints=data.get("endpoints", {}),
            security_level=data.get("security_level", "standard"),
            owner=data.get("owner", ""),
            contact=data.get("contact", ""),
        )


@dataclass
class A2ATask:
    task_id: str
    method: str
    params: dict
    sender_card: AgentCard
    priority: str = "normal"  # low | normal | high | critical
    deadline_ms: int = 60000
    require_approval: bool = False

    def to_jsonrpc(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self.task_id,
            "method": self.method,
            "params": {
                "task": self.params,
                "agent": self.sender_card.to_dict(),
                "priority": self.priority,
                "deadline_ms": self.deadline_ms,
            },
        }


@dataclass
class A2AResult:
    task_id: str
    success: bool
    result: dict = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    responded_by: str = ""


class A2AProtocol:
    """Agent-to-Agent communication protocol handler.

    Usage:
        a2a = A2AProtocol(AgentCard(
            name="AI-Agent-Playground",
            capabilities=["code_execution", "web_search"],
            tools=["calculator", "run_python", "read_file"],
        ))
        a2a.register_handler("tasks/send", handle_task)
        result = await a2a.send_task(
            to="http://other-agent:8080/a2a",
            method="tasks/send",
            params={"query": "Analyze this code"},
        )
    """

    def __init__(self, agent_card: AgentCard):
        self.card = agent_card
        self.handlers: dict[str, callable] = {}
        self._known_agents: dict[str, AgentCard] = {}
        self._task_history: list[dict] = []
        A2A_DIR.mkdir(parents=True, exist_ok=True)

    def register_handler(self, method: str, handler: callable):
        """Register a handler for a JSON-RPC method."""
        self.handlers[method] = handler
        logger.info("A2A: registered handler for '%s'", method)

    def has_capability(self, capability: str) -> bool:
        return capability in self.card.capabilities

    def matches_task(self, task: dict) -> bool:
        """Check if this agent can handle a given task."""
        required_caps = task.get("required_capabilities", [])
        if required_caps:
            return all(c in self.card.capabilities for c in required_caps)

        required_tools = task.get("required_tools", [])
        if required_tools:
            return all(t in self.card.tools for t in required_tools)

        return any(
            c in task.get("description", "") for c in self.card.capabilities
        ) if self.card.capabilities else True

    def register_agent(self, agent_card: AgentCard):
        """Add a known agent to the registry."""
        agent_id = hashlib.sha256(
            f"{agent_card.name}{''.join(agent_card.capabilities)}".encode()
        ).hexdigest()[:12]
        self._known_agents[agent_id] = agent_card
        self._save_registry()
        return agent_id

    def unregister_agent(self, agent_id: str):
        self._known_agents.pop(agent_id, None)
        self._save_registry()

    def _save_registry(self):
        path = A2A_DIR / "agent_registry.json"
        data = {aid: a.to_dict() for aid, a in self._known_agents.items()}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_registry(self):
        path = A2A_DIR / "agent_registry.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._known_agents = {aid: AgentCard.from_dict(card) for aid, card in data.items()}

    async def handle_request(self, request: dict) -> A2AResult:
        """Process an incoming A2A JSON-RPC request."""
        task_id = request.get("id", "unknown")
        method = request.get("method", "")
        params = request.get("params", {})
        sender_data = params.get("agent", {})

        t0 = time.time()
        sender = AgentCard.from_dict(sender_data) if sender_data else None

        # Register sender
        if sender:
            sid = self.register_agent(sender)
            logger.info("A2A: received task from %s (id=%s)", sender.name, sid)

        handler = self.handlers.get(method)
        if not handler:
            return A2AResult(
                task_id=task_id, success=False,
                error=f"Method not found: {method}",
                latency_ms=(time.time() - t0) * 1000,
                responded_by=self.card.name,
            )

        # Security: validate sender
        if sender and sender.security_level == "critical":
            if self.card.security_level in ("standard",):
                logger.warning("A2A: rejected critical agent %s (our level: %s)",
                             sender.name, self.card.security_level)
                return A2AResult(
                    task_id=task_id, success=False,
                    error="Insufficient security level to handle critical agent tasks",
                    latency_ms=(time.time() - t0) * 1000,
                    responded_by=self.card.name,
                )

        try:
            result = await handler(task_id, params.get("task", {}), sender)
            return A2AResult(
                task_id=task_id, success=True, result=result,
                latency_ms=(time.time() - t0) * 1000,
                responded_by=self.card.name,
            )
        except Exception as e:
            return A2AResult(
                task_id=task_id, success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
                responded_by=self.card.name,
            )

    def list_agents(self) -> list[dict]:
        return [a.to_dict() for a in self._known_agents.values()]

    def status(self) -> dict:
        return {
            "agent": self.card.to_dict(),
            "handlers": list(self.handlers.keys()),
            "known_agents": len(self._known_agents),
            "task_history": len(self._task_history),
        }
