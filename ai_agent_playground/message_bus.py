"""Message Bus — Agent 之间的统一通信层。"""

import hashlib
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from weakref import WeakSet


@dataclass
class Message:
    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""
    source: str = ""


class MessageDeduplicator:
    def __init__(self, ttl_ms: float = 1000):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl_ms = ttl_ms

    def is_duplicate(self, topic: str, payload: Any) -> bool:
        key = self._make_key(topic, payload)
        now = time.time() * 1000
        if key in self._cache:
            old_payload, old_time = self._cache[key]
            if now - old_time < self._ttl_ms and self._payload_equal(old_payload, payload):
                return True
        self._cache[key] = (payload, now)
        self._cleanup(now)
        return False

    def _make_key(self, topic: str, payload: Any) -> str:
        content = json.dumps({"topic": topic, "payload": payload}, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _payload_equal(self, a: Any, b: Any) -> bool:
        try:
            return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return a == b

    def _cleanup(self, now: float):
        expired = [k for k, (_, t) in self._cache.items() if now - t > self._ttl_ms * 2]
        for k in expired:
            self._cache.pop(k, None)


class IncrementalSync:
    def __init__(self):
        self._last_state: dict[str, dict] = defaultdict(dict)

    def get_delta(self, agent_id: str, new_state: dict) -> dict:
        old_state = self._last_state.get(agent_id, {})
        delta = {}
        for key, value in new_state.items():
            if key not in old_state or old_state[key] != value:
                delta[key] = value
        self._last_state[agent_id] = new_state.copy()
        return delta


class MessageBatcher:
    def __init__(self, max_size: int = 10, flush_interval_ms: float = 100):
        self._buffer: list[Message] = []
        self._max_size = max_size
        self._flush_interval_ms = flush_interval_ms
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def add(self, message: Message) -> list[Message] | None:
        with self._lock:
            self._buffer.append(message)
            if len(self._buffer) >= self._max_size or (time.time() - self._last_flush) * 1000 >= self._flush_interval_ms:
                batch = self._buffer.copy()
                self._buffer.clear()
                self._last_flush = time.time()
                return batch
        return None


class AgentLocalCache:
    def __init__(self, default_ttl_ms: float = 5000):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl_ms = default_ttl_ms
        self._lock = threading.RLock()

    def get_or_fetch(self, key: str, fetcher: Callable[[], Any], ttl_ms: float | None = None) -> Any:
        with self._lock:
            now = time.time() * 1000
            ttl = ttl_ms or self._default_ttl_ms
            if key in self._cache:
                value, expiry = self._cache[key]
                if now < expiry:
                    return value
            value = fetcher()
            self._cache[key] = (value, now + ttl)
            return value

    def get(self, key: str) -> Any | None:
        with self._lock:
            now = time.time() * 1000
            if key in self._cache:
                value, expiry = self._cache[key]
                if now < expiry:
                    return value
                del self._cache[key]
            return None

    def set(self, key: str, value: Any, ttl_ms: float | None = None):
        with self._lock:
            now = time.time() * 1000
            self._cache[key] = (value, now + (ttl_ms or self._default_ttl_ms))


class MessageBus:
    def __init__(self, enable_dedup: bool = True, enable_batch: bool = True, dedup_ttl_ms: float = 1000, batch_size: int = 10, batch_interval_ms: float = 100):
        self._subscribers: dict[str, WeakSet[Callable]] = defaultdict(WeakSet)
        self._lock = threading.RLock()
        self._dedup = MessageDeduplicator(ttl_ms=dedup_ttl_ms) if enable_dedup else None
        self._incremental = IncrementalSync()
        self._batcher = MessageBatcher(max_size=batch_size, flush_interval_ms=batch_interval_ms) if enable_batch else None
        self._agent_caches: dict[str, AgentLocalCache] = {}
        self._stats = {"published": 0, "delivered": 0, "deduped": 0, "batched": 0, "errors": 0}

    def subscribe(self, topic: str, handler: Callable):
        with self._lock:
            self._subscribers[topic].add(handler)

    def unsubscribe(self, topic: str, handler: Callable):
        with self._lock:
            self._subscribers[topic].discard(handler)

    def publish(self, topic: str, payload: Any, source: str = "", trace_id: str = "", use_delta: bool = False, agent_id: str = "") -> bool:
        try:
            if use_delta and agent_id:
                payload = self._incremental.get_delta(agent_id, payload)
            if self._dedup and self._dedup.is_duplicate(topic, payload):
                self._stats["deduped"] += 1
                return False
            message = Message(topic=topic, payload=payload, source=source, trace_id=trace_id)
            if self._batcher:
                batch = self._batcher.add(message)
                if batch:
                    self._deliver_batch(batch)
                    self._stats["batched"] += 1
            else:
                self._deliver(message)
            self._stats["published"] += 1
            return True
        except Exception as e:
            self._stats["errors"] += 1
            print(f"[MessageBus] Publish error: {e}")
            return False

    def _deliver(self, message: Message):
        with self._lock:
            handlers = self._subscribers.get(message.topic, WeakSet()).copy()
        for handler in handlers:
            try:
                handler(message.payload)
                self._stats["delivered"] += 1
            except Exception as e:
                self._stats["errors"] += 1

    def _deliver_batch(self, batch: list[Message]):
        for message in batch:
            self._deliver(message)

    def get_cache(self, agent_id: str) -> AgentLocalCache:
        with self._lock:
            if agent_id not in self._agent_caches:
                self._agent_caches[agent_id] = AgentLocalCache()
            return self._agent_caches[agent_id]

    def get_stats(self) -> dict:
        return self._stats.copy()

    def print_stats(self):
        s = self._stats
        print(f"[MessageBus] published={s['published']} delivered={s['delivered']} deduped={s['deduped']} batched={s['batched']} errors={s['errors']}")


_message_bus: MessageBus | None = None


def get_message_bus() -> MessageBus:
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


def subscribe(topic: str):
    def decorator(func: Callable):
        get_message_bus().subscribe(topic, func)
        return func
    return decorator


def publish(topic: str, payload: Any, **kwargs):
    get_message_bus().publish(topic, payload, **kwargs)


message_bus = get_message_bus()