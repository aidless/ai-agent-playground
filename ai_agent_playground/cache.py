"""LLM Cache — LLM响应缓存。"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 3600


class LLMCache:
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    def _make_key(self, messages: list, model: str, **kwargs) -> str:
        content = json.dumps({"messages": messages, "model": model, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, messages: list, model: str, **kwargs) -> str | None:
        key = self._make_key(messages, model, **kwargs)
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            entry = self._cache[key]
            now = time.time()
            if now - entry.created_at > entry.ttl:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            entry.accessed_at = now
            entry.access_count += 1
            self._stats["hits"] += 1
            return entry.value

    def set(self, messages: list, model: str, response: str, ttl: float | None = None):
        key = self._make_key(messages, model)
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict()
            self._cache[key] = CacheEntry(key=key, value=response, ttl=ttl or self.default_ttl)
            self._stats["sets"] += 1

    def _evict(self):
        if not self._cache:
            return
        oldest = min(self._cache.values(), key=lambda e: e.accessed_at)
        del self._cache[oldest.key]
        self._stats["evictions"] += 1

    def invalidate(self, pattern: str | None = None):
        with self._lock:
            if pattern:
                to_delete = [k for k in self._cache if pattern in k]
                for k in to_delete:
                    del self._cache[k]
            else:
                self._cache.clear()

    def get_stats(self) -> dict:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {"hits": self._stats["hits"], "misses": self._stats["misses"], "hit_rate": f"{hit_rate*100:.1f}%", "size": len(self._cache), "evictions": self._stats["evictions"]}

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n=== LLM Cache ===")
        print(f"Hits: {stats['hits']} | Misses: {stats['misses']} | Hit rate: {stats['hit_rate']}")
        print(f"Size: {stats['size']} | Evictions: {stats['evictions']}")


class EmbeddingCache:
    def __init__(self, max_size: int = 5000, default_ttl: float = 86400):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def _make_key(self, text: str, model: str = "default") -> str:
        content = json.dumps({"text": text, "model": model}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str = "default") -> list[float] | None:
        key = self._make_key(text, model)
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if time.time() - entry.created_at > entry.ttl:
                del self._cache[key]
                return None
            entry.accessed_at = time.time()
            entry.access_count += 1
            return entry.value

    def set(self, text: str, embedding: list[float], model: str = "default", ttl: float | None = None):
        key = self._make_key(text, model)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest = min(self._cache.values(), key=lambda e: e.accessed_at)
                del self._cache[oldest.key]
            self._cache[key] = CacheEntry(key=key, value=embedding, ttl=ttl or self.default_ttl)


_llm_cache: LLMCache | None = None
_embedding_cache: EmbeddingCache | None = None


def get_llm_cache() -> LLMCache:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMCache()
    return _llm_cache


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache