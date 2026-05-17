"""Extension System — 配置化 + 插件体系 + 动态加载。"""

import importlib
import importlib.util
import json
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class AgentConfig:
    name: str
    class_path: str
    config: dict = field(default_factory=dict)
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    agents: list[AgentConfig] = field(default_factory=list)
    message_bus: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)
    resilience: dict = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
        content = Path(path).read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}
    except ImportError:
        return _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))


def load_json(path: str | Path) -> dict:
    content = Path(path).read_text(encoding="utf-8")
    return json.loads(content)


def load_config(path: str | Path) -> SystemConfig:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = load_yaml(path)
    elif suffix == ".json":
        data = load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    agents = []
    for a in data.get("agents", []):
        agents.append(AgentConfig(name=a["name"], class_path=a["class_path"], config=a.get("config", {}), enabled=a.get("enabled", True), dependencies=a.get("dependencies", [])))
    return SystemConfig(agents=agents, message_bus=data.get("message_bus", {}), observability=data.get("observability", {}), resilience=data.get("resilience", {}))


def _parse_simple_yaml(content: str) -> dict:
    result = {}
    current_section = None
    current_items = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith(" "):
            if current_section and current_items:
                result[current_section] = current_items
            current_section = line.rstrip(":")
            current_items = []
        elif ":" in line and current_section:
            key, value = line.split(":", 1)
            current_items.append({key.strip(): value.strip()})
    if current_section and current_items:
        result[current_section] = current_items
    return result


def register_from_config(config: SystemConfig):
    from ai_agent_playground.agent_registry import register
    for agent_config in config.agents:
        if not agent_config.enabled:
            continue
        cls = _import_class(agent_config.class_path)
        if cls is None:
            print(f"[Extension] Failed to import: {agent_config.class_path}")
            continue
        register(agent_config.name, cls)
        print(f"[Extension] Registered: {agent_config.name}")


def _import_class(class_path: str) -> type | None:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"[Extension] Import error: {e}")
        return None


class PluginManager:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self._loaded_plugins: dict[str, type] = {}
        self._lock = threading.RLock()

    def load_plugin(self, plugin_path: str | Path) -> bool:
        plugin_path = Path(plugin_path)
        try:
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_path.stem] = module
                spec.loader.exec_module(module)
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and hasattr(obj, "run") and name.endswith("Agent"):
                        with self._lock:
                            self._loaded_plugins[name] = obj
                        print(f"[PluginManager] Loaded: {name}")
                        return True
            return False
        except Exception as e:
            print(f"[PluginManager] Load error: {e}")
            return False

    def load_all(self, pattern: str = "*.py") -> int:
        if not self.plugin_dir.exists():
            return 0
        count = 0
        for path in self.plugin_dir.glob(pattern):
            if path.name.startswith("_"):
                continue
            if self.load_plugin(path):
                count += 1
        return count

    def get_plugin(self, name: str) -> type | None:
        return self._loaded_plugins.get(name)

    def list_plugins(self) -> list[str]:
        return list(self._loaded_plugins.keys())


class WorkerPool:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._workers: list[threading.Thread] = []
        self._task_queue: list[tuple[Callable, tuple, dict]] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
        print(f"[WorkerPool] Started {self.num_workers} workers")

    def stop(self):
        self._running = False
        for worker in self._workers:
            worker.join(timeout=5)
        self._workers.clear()
        print("[WorkerPool] Stopped")

    def submit(self, func: Callable, *args, **kwargs):
        with self._lock:
            self._task_queue.append((func, args, kwargs))

    def _worker_loop(self):
        while self._running:
            task = None
            with self._lock:
                if self._task_queue:
                    task = self._task_queue.pop(0)
            if task:
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"[WorkerPool] Task error: {e}")
            else:
                import time
                time.sleep(0.1)

    def get_stats(self) -> dict:
        return {"num_workers": self.num_workers, "pending_tasks": len(self._task_queue), "running": self._running}


_plugin_manager: PluginManager | None = None
_worker_pool: WorkerPool | None = None


def get_plugin_manager(plugin_dir: str = "plugins") -> PluginManager:
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(plugin_dir=plugin_dir)
    return _plugin_manager


def get_worker_pool(num_workers: int = 4) -> WorkerPool:
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = WorkerPool(num_workers=num_workers)
    return _worker_pool