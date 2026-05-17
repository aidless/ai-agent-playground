"""Agent Registry — Agent注册中心，支持动态注册/发现。"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentMetadata:
    name: str
    agent_class: type
    instance: Any = None
    registered_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    call_count: int = 0
    error_count: int = 0
    status: str = "ready"


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentMetadata] = {}
        self._lock = threading.RLock()
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, agent_class: type | None = None, *, factory: Callable | None = None) -> Callable:
        def decorator(cls: type):
            with self._lock:
                self._agents[name] = AgentMetadata(name=name, agent_class=cls)
                self._factories[name] = factory or (lambda: cls())
            return cls

        if agent_class is not None:
            with self._lock:
                self._agents[name] = AgentMetadata(name=name, agent_class=agent_class)
                self._factories[name] = factory or (lambda: agent_class())
            return agent_class
        return decorator

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._agents:
                self._agents[name].status = "stopped"
                return True
            return False

    def get(self, name: str, auto_instantiate: bool = True) -> Any:
        with self._lock:
            if name not in self._agents:
                raise KeyError(f"Agent not found: {name}")
            meta = self._agents[name]
            if meta.instance is None and auto_instantiate:
                if name in self._factories:
                    meta.instance = self._factories[name]()
                else:
                    meta.instance = meta.agent_class()
            meta.last_used = time.time()
            meta.call_count += 1
            return meta.instance

    def get_metadata(self, name: str) -> AgentMetadata | None:
        with self._lock:
            return self._agents.get(name)

    def list_all(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    def list_by_status(self, status: str) -> list[str]:
        with self._lock:
            return [name for name, meta in self._agents.items() if meta.status == status]

    def update_status(self, name: str, status: str):
        with self._lock:
            if name in self._agents:
                self._agents[name].status = status

    def record_error(self, name: str):
        with self._lock:
            if name in self._agents:
                self._agents[name].error_count += 1
                if self._agents[name].error_count >= 3:
                    self._agents[name].status = "error"

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total": len(self._agents),
                "ready": sum(1 for m in self._agents.values() if m.status == "ready"),
                "busy": sum(1 for m in self._agents.values() if m.status == "busy"),
                "error": sum(1 for m in self._agents.values() if m.status == "error"),
                "total_calls": sum(m.call_count for m in self._agents.values()),
                "total_errors": sum(m.error_count for m in self._agents.values()),
            }

    def print_dashboard(self):
        stats = self.get_stats()
        print("\n=== Agent Registry ===")
        print(f"Total: {stats['total']} | Ready: {stats['ready']} | Busy: {stats['busy']} | Error: {stats['error']}")
        print(f"Total calls: {stats['total_calls']} | Errors: {stats['total_errors']}")
        print("\n--- Agents ---")
        with self._lock:
            for name, meta in self._agents.items():
                status_icon = {"ready": "✅", "busy": "🔄", "error": "❌", "stopped": "⏹"}.get(meta.status, "❓")
                print(f"  {status_icon} {name}: calls={meta.call_count} errors={meta.error_count}")


_agent_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


def register(name: str, agent_class: type | None = None, **kwargs):
    return get_agent_registry().register(name, agent_class, **kwargs)


def get_agent(name: str) -> Any:
    return get_agent_registry().get(name)


def list_agents() -> list[str]:
    return get_agent_registry().list_all()


agent_registry = get_agent_registry()