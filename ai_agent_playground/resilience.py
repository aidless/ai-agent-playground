"""Resilience — 错误恢复机制。"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar


T = TypeVar("T")


def retry(max_attempts: int = 3, base_delay: float = 1.0, exponential: bool = True, max_delay: float = 60.0, exceptions: tuple = (Exception,)):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as last_exception:
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay) if exponential else base_delay
                        print(f"[Retry] {func.__name__} failed (attempt {attempt+1}/{max_attempts}), retrying in {delay:.1f}s: {last_exception}")
                        time.sleep(delay)
                    else:
                        print(f"[Retry] {func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        return wrapper
    return decorator


@dataclass
class CircuitState:
    name: str
    status: str = "closed"
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    opened_at: float = 0


class CircuitBreaker:
    def __init__(self, name: str = "default", failure_threshold: int = 3, success_threshold: int = 2, recovery_timeout: float = 30.0, half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState(name=name)
        self._lock = threading.RLock()
        self._half_open_calls = 0

    @property
    def status(self) -> str:
        with self._lock:
            if self._state.status == "open":
                if time.time() - self._state.opened_at >= self.recovery_timeout:
                    return "half_open"
            return self._state.status

    def can_execute(self) -> bool:
        with self._lock:
            status = self.status
            if status == "closed":
                return True
            if status == "open":
                return False
            return self._half_open_calls < self.half_open_max_calls

    def record_success(self):
        with self._lock:
            if self._state.status == "half_open":
                self._state.success_count += 1
                if self._state.success_count >= self.success_threshold:
                    print(f"[CircuitBreaker] {self.name} CLOSED (recovered)")
                    self._state.status = "closed"
                    self._state.failure_count = 0
                    self._state.success_count = 0
            elif self._state.status == "closed":
                self._state.failure_count = max(0, self._state.failure_count - 1)

    def record_failure(self):
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            if self._state.status == "closed":
                if self._state.failure_count >= self.failure_threshold:
                    print(f"[CircuitBreaker] {self.name} OPEN (failure threshold reached)")
                    self._state.status = "open"
                    self._state.opened_at = time.time()
            elif self._state.status == "half_open":
                print(f"[CircuitBreaker] {self.name} OPEN (half_open failed)")
                self._state.status = "open"
                self._state.opened_at = time.time()
                self._half_open_calls = 0

    def __enter__(self):
        if not self.can_execute():
            raise CircuitOpenError(f"CircuitBreaker '{self.name}' is OPEN")
        with self._lock:
            if self.status == "half_open":
                self._half_open_calls += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False


class CircuitOpenError(Exception):
    pass


def with_timeout(func: Callable[..., T], timeout: float, default: T | None = None) -> T:
    result = [default]
    exception = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return default
    if exception[0]:
        raise exception[0]
    return result[0]


class CircuitBreakerManager:
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get(self, name: str, **kwargs) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]

    def get_all_status(self) -> dict:
        with self._lock:
            return {name: cb.status for name, cb in self._breakers.items()}

    def print_dashboard(self):
        print("\n=== Circuit Breakers ===")
        with self._lock:
            if not self._breakers:
                print("  (none)")
                return
            for name, cb in self._breakers.items():
                status = cb.status
                icon = {"closed": "✅", "open": "🔴", "half_open": "🟡"}.get(status, "❓")
                print(f"  {icon} {name}: {status} (failures={cb._state.failure_count})")


_cb_manager: CircuitBreakerManager | None = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    global _cb_manager
    if _cb_manager is None:
        _cb_manager = CircuitBreakerManager()
    return _cb_manager