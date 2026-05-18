"""Multi-Tenant Isolation — namespace, quota, and isolation for agent workloads.

Each tenant gets:
  - Isolated working directory
  - Rate limit / quota enforcement
  - Separate audit trail
  - Tenant-level identity and permissions
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TenantQuota:
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    max_concurrent_sessions: int = 5
    max_tokens_per_day: int = 1_000_000
    storage_mb: int = 100


@dataclass
class Tenant:
    id: str
    name: str
    namespace: str
    quota: TenantQuota = field(default_factory=TenantQuota)
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def work_dir(self) -> Path:
        return Path("./tenant_workspaces") / self.namespace

    @property
    def audit_dir(self) -> Path:
        return self.work_dir / "audit"


class TenancyManager:
    """Manages tenant isolation and quotas.

    Usage:
        tm = TenancyManager()
        tm.register_tenant("org-a", "Org Alpha")
        tm.check_quota("org-a", "read_file")  # Raises if exceeded
        tm.record_usage("org-a", "read_file")
    """

    def __init__(self, store_path: str = "./tenant_registry"):
        self.store = Path(store_path)
        self.store.mkdir(parents=True, exist_ok=True)
        self._tenants: dict[str, Tenant] = {}
        self._usage: dict[str, dict[str, list[float]]] = {}  # tenant -> {request_timestamps}
        self._sessions: dict[str, int] = {}  # tenant -> active_session_count
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        reg_file = self.store / "tenants.json"
        if reg_file.exists():
            data = json.loads(reg_file.read_text(encoding="utf-8"))
            for t in data.get("tenants", []):
                quota = TenantQuota(**t.get("quota", {}))
                self._tenants[t["id"]] = Tenant(
                    id=t["id"], name=t["name"], namespace=t.get("namespace", t["id"]),
                    quota=quota, active=t.get("active", True),
                    created_at=t.get("created_at", ""),
                )
                self._tenants[t["id"]].work_dir.mkdir(parents=True, exist_ok=True)
                self._tenants[t["id"]].audit_dir.mkdir(parents=True, exist_ok=True)

    def _save(self):
        data = {
            "tenants": [
                {
                    "id": t.id, "name": t.name, "namespace": t.namespace,
                    "quota": {
                        "requests_per_minute": t.quota.requests_per_minute,
                        "requests_per_hour": t.quota.requests_per_hour,
                        "max_concurrent_sessions": t.quota.max_concurrent_sessions,
                        "max_tokens_per_day": t.quota.max_tokens_per_day,
                        "storage_mb": t.quota.storage_mb,
                    },
                    "active": t.active, "created_at": t.created_at,
                }
                for t in self._tenants.values()
            ]
        }
        (self.store / "tenants.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def register_tenant(self, tenant_id: str, name: str, quota: TenantQuota | None = None) -> Tenant:
        with self._lock:
            t = Tenant(
                id=tenant_id, name=name, namespace=tenant_id,
                quota=quota or TenantQuota(),
            )
            t.work_dir.mkdir(parents=True, exist_ok=True)
            t.audit_dir.mkdir(parents=True, exist_ok=True)
            self._tenants[tenant_id] = t
            self._usage[tenant_id] = {}
            self._save()
            return t

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def check_quota(self, tenant_id: str, tool_name: str) -> dict:
        """Check if a request would exceed quota. Returns {allowed: bool, reason: str}."""
        tenant = self._tenants.get(tenant_id)
        if not tenant or not tenant.active:
            return {"allowed": False, "reason": "tenant not found or inactive"}
        if self._sessions.get(tenant_id, 0) >= tenant.quota.max_concurrent_sessions:
            return {"allowed": False, "reason": "max concurrent sessions exceeded"}

        now = time.time()
        with self._lock:
            ts_list = self._usage.setdefault(tenant_id, {}).setdefault(tool_name, [])
            # Clean old entries
            ts_list[:] = [t for t in ts_list if now - t < 3600]

            rpm = sum(1 for t in ts_list if now - t < 60)
            if rpm >= tenant.quota.requests_per_minute:
                return {"allowed": False, "reason": f"rate limit: {rpm}/{tenant.quota.requests_per_minute} rpm"}

            rph = len(ts_list)
            if rph >= tenant.quota.requests_per_hour:
                return {"allowed": False, "reason": f"rate limit: {rph}/{tenant.quota.requests_per_hour} rph"}

        return {"allowed": True, "reason": "ok"}

    def record_usage(self, tenant_id: str, tool_name: str):
        """Record a successful request for quota tracking."""
        with self._lock:
            self._usage.setdefault(tenant_id, {}).setdefault(tool_name, []).append(time.time())

    def session_start(self, tenant_id: str):
        with self._lock:
            self._sessions[tenant_id] = self._sessions.get(tenant_id, 0) + 1

    def session_end(self, tenant_id: str):
        with self._lock:
            self._sessions[tenant_id] = max(0, self._sessions.get(tenant_id, 0) - 1)

    def get_tenant_storage_usage_mb(self, tenant_id: str) -> float:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return 0.0
        total = 0
        for f in tenant.work_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)

    def list_tenants(self) -> list[dict]:
        return [
            {
                "id": t.id, "name": t.name, "active": t.active,
                "sessions": self._sessions.get(t.id, 0),
                "storage_mb": round(self.get_tenant_storage_usage_mb(t.id), 2),
            }
            for t in self._tenants.values()
        ]

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and their data."""
        with self._lock:
            if tenant_id not in self._tenants:
                return False
            tenant = self._tenants.pop(tenant_id)
            self._usage.pop(tenant_id, None)
            self._sessions.pop(tenant_id, None)
            import shutil
            if tenant.work_dir.exists():
                shutil.rmtree(str(tenant.work_dir), ignore_errors=True)
            self._save()
            return True
