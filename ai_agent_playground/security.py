"""Security — Agent安全控制。"""

import hashlib
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Permission:
    principal: str
    resource: str
    granted_at: float = field(default_factory=time.time)
    expires_at: float = 0


class PermissionManager:
    def __init__(self):
        self._permissions: dict[str, list[Permission]] = defaultdict(list)
        self._roles: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def grant(self, principal: str, resource: str, expires_at: float = 0):
        with self._lock:
            perm = Permission(principal=principal, resource=resource, expires_at=expires_at)
            self._permissions[principal].append(perm)

    def revoke(self, principal: str, resource: str) -> bool:
        with self._lock:
            perms = self._permissions.get(principal, [])
            for i, p in enumerate(perms):
                if p.resource == resource:
                    perms.pop(i)
                    return True
            return False

    def check(self, principal: str, resource: str) -> bool:
        with self._lock:
            now = time.time()
            perms = self._permissions.get(principal, [])
            for p in perms:
                if p.resource == resource:
                    if p.expires_at == 0 or p.expires_at > now:
                        return True
            return False

    def assign_role(self, principal: str, role: str):
        with self._lock:
            self._roles[role].add(principal)

    def get_role_permissions(self, role: str) -> list[str]:
        role_perms = {"admin": ["*"], "developer": ["planner", "executor", "reviewer"], "viewer": ["reader"]}
        return role_perms.get(role, [])

    def check_with_role(self, principal: str, resource: str) -> bool:
        if self.check(principal, resource):
            return True
        with self._lock:
            for role, members in self._roles.items():
                if principal in members:
                    perms = self.get_role_permissions(role)
                    if "*" in perms or resource in perms:
                        return True
        return False


class InputValidator:
    DANGEROUS_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|commands?)",
        r"(system|admin)\s*:\s*",
        r"<\s*/?script",
        r"\{\{.*\}\}",
        r"\$\{.*\}",
        r"<!--.*-->",
        r"<iframe",
        r"javascript:",
        r"on\s*=\s*",
    ]
    SENSITIVE_KEYWORDS = ["password", "secret", "api_key", "token", "credential", "sudo", "rm -rf", "delete", "drop", "truncate"]

    def __init__(self):
        self._dangerous_regex = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
        self._stats = {"validated": 0, "rejected": 0, "warnings": 0}

    def validate(self, user_input: str) -> tuple[bool, str]:
        self._stats["validated"] += 1
        if not user_input:
            return False, "Empty input"
        for pattern in self._dangerous_regex:
            if pattern.search(user_input):
                self._stats["rejected"] += 1
                return False, f"Dangerous pattern detected: {pattern.pattern}"
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword.lower() in user_input.lower():
                self._stats["warnings"] += 1
        if len(user_input) > 100000:
            self._stats["rejected"] += 1
            return False, "Input too long"
        return True, ""

    def sanitize(self, user_input: str) -> str:
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", user_input)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def get_stats(self) -> dict:
        return self._stats.copy()


@dataclass
class RateLimitRecord:
    count: int = 0
    window_start: float = field(default_factory=time.time)


class RateLimiter:
    def __init__(self, max_requests: int = 100, window_sec: float = 60, block_duration: float = 300):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.block_duration = block_duration
        self._records: dict[str, RateLimitRecord] = {}
        self._blocked: dict[str, float] = {}
        self._lock = threading.RLock()

    def check(self, key: str) -> tuple[bool, str]:
        now = time.time()
        with self._lock:
            if key in self._blocked:
                if now < self._blocked[key]:
                    remaining = int(self._blocked[key] - now)
                    return False, f"Blocked for {remaining}s"
                else:
                    del self._blocked[key]
            if key not in self._records:
                self._records[key] = RateLimitRecord()
            record = self._records[key]
            if now - record.window_start > self.window_sec:
                record.count = 0
                record.window_start = now
            if record.count >= self.max_requests:
                self._blocked[key] = now + self.block_duration
                return False, "Rate limit exceeded, blocked"
            record.count += 1
            return True, ""

    def get_remaining(self, key: str) -> int:
        with self._lock:
            record = self._records.get(key)
            if not record:
                return self.max_requests
            return max(0, self.max_requests - record.count)


_permission_manager: PermissionManager | None = None
_input_validator: InputValidator | None = None
_rate_limiter: RateLimiter | None = None


def get_permission_manager() -> PermissionManager:
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


def get_input_validator() -> InputValidator:
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_rate_limiter(max_requests: int = 100, window_sec: float = 60) -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests=max_requests, window_sec=window_sec)
    return _rate_limiter