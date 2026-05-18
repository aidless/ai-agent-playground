"""Deployment Manager — canary release, semantic versioning, rollback.

Enterprise deployment patterns:
  - Canary: route X% of traffic to new version, monitor, expand or rollback
  - Blue/Green: two identical environments, swap on release
  - Rolling: gradually replace instances

This module implements a lightweight deployment manager suitable for
single-service agents. For multi-service, integrate with K8s/Helm.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class DeployEnv(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"


class ReleaseStatus(str, Enum):
    PENDING = "pending"
    ROLLING_OUT = "rolling_out"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


@dataclass
class Version:
    major: int = 0
    minor: int = 1
    patch: int = 0
    build: str = ""

    def __str__(self):
        base = f"v{self.major}.{self.minor}.{self.patch}"
        return f"{base}+{self.build}" if self.build else base

    def bump_patch(self):
        return Version(self.major, self.minor, self.patch + 1)

    def bump_minor(self):
        return Version(self.major, self.minor + 1, 0)

    def bump_major(self):
        return Version(self.major + 1, 0, 0)

    @classmethod
    def parse(cls, s: str):
        s = s.lstrip("v")
        build = ""
        if "+" in s:
            s, build = s.split("+", 1)
        parts = [int(x) for x in s.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return cls(parts[0], parts[1], parts[2], build)


@dataclass
class Release:
    version: Version
    status: ReleaseStatus = ReleaseStatus.PENDING
    deployed_at: str = ""
    deployed_by: str = ""
    env: str = "dev"
    traffic_pct: int = 0
    commit_hash: str = ""
    rollback_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self):
        return self.status == ReleaseStatus.ACTIVE


class DeploymentManager:
    """Manages releases across environments with canary support.

    Usage:
        dm = DeploymentManager()
        dm.deploy("production", traffic_pct=10)   # Start canary
        dm.promote_canary()                        # Expand to 100%
        dm.rollback("production")                  # Rollback on failure
    """

    def __init__(self, store_path: str = "./deploy_history"):
        self.store = Path(store_path)
        self.store.mkdir(parents=True, exist_ok=True)
        self._current: dict[str, Release] = {}  # env -> current release
        self._history: list[Release] = []
        self._version_file = self.store / "version.json"
        self._load()

    def _load(self):
        if self._version_file.exists():
            data = json.loads(self._version_file.read_text(encoding="utf-8"))
            for env, rel_data in data.get("current", {}).items():
                self._current[env] = Release(
                    version=Version.parse(rel_data["version"]),
                    status=ReleaseStatus(rel_data.get("status", "active")),
                    deployed_at=rel_data.get("deployed_at", ""),
                    env=env,
                    traffic_pct=rel_data.get("traffic_pct", 100),
                    commit_hash=rel_data.get("commit_hash", ""),
                )
            for r in data.get("history", []):
                self._history.append(Release(
                    version=Version.parse(r["version"]),
                    status=ReleaseStatus(r.get("status", "active")),
                    deployed_at=r.get("deployed_at", ""),
                    deployed_by=r.get("deployed_by", ""),
                    env=r.get("env", ""),
                    traffic_pct=r.get("traffic_pct", 100),
                    commit_hash=r.get("commit_hash", ""),
                    rollback_to=r.get("rollback_to", ""),
                ))

    def _save(self):
        data = {
            "current": {
                env: {
                    "version": str(r.version),
                    "status": r.status.value,
                    "deployed_at": r.deployed_at,
                    "traffic_pct": r.traffic_pct,
                    "commit_hash": r.commit_hash,
                }
                for env, r in self._current.items()
            },
            "history": [
                {
                    "version": str(r.version),
                    "status": r.status.value,
                    "deployed_at": r.deployed_at,
                    "env": r.env,
                    "traffic_pct": r.traffic_pct,
                    "commit_hash": r.commit_hash,
                }
                for r in self._history[-50:]  # Keep last 50
            ],
        }
        self._version_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def current_version(self, env: str = "production") -> Version | None:
        rel = self._current.get(env)
        return rel.version if rel else None

    def deploy(
        self,
        env: str = "staging",
        traffic_pct: int = 100,
        commit_hash: str = "",
        deployed_by: str = "ci",
    ) -> Release:
        """Deploy a new version to an environment."""
        current = self._current.get(env)
        new_version = (
            current.version.bump_patch()
            if current else Version(0, 1, 0)
        )

        release = Release(
            version=new_version,
            status=ReleaseStatus.ROLLING_OUT,
            deployed_at=datetime.now(timezone.utc).isoformat(),
            deployed_by=deployed_by,
            env=env,
            traffic_pct=traffic_pct,
            commit_hash=commit_hash,
        )
        self._current[env] = release
        self._history.append(release)
        self._save()

        # If previous release exists, mark superseded
        if current and current.status == ReleaseStatus.ACTIVE:
            current.status = ReleaseStatus.SUPERSEDED

        return release

    def promote_canary(self, env: str = "production") -> Release | None:
        """Promote canary from X% to 100% traffic."""
        current = self._current.get(env)
        if not current or current.traffic_pct >= 100:
            return None

        current.traffic_pct = 100
        current.status = ReleaseStatus.ACTIVE
        self._save()
        return current

    def rollback(self, env: str = "production") -> Release | None:
        """Rollback to the previous version in history."""
        current = self._current.get(env)
        if not current:
            return None

        # Find the version before the current one
        env_releases = [r for r in self._history if r.env == env]
        if len(env_releases) < 2:
            return None

        prev = env_releases[-2]
        current.status = ReleaseStatus.ROLLED_BACK
        current.rollback_to = str(prev.version)

        rollback_release = Release(
            version=prev.version,
            status=ReleaseStatus.ACTIVE,
            deployed_at=datetime.now(timezone.utc).isoformat(),
            deployed_by="auto-rollback",
            env=env,
            traffic_pct=100,
            commit_hash=prev.commit_hash,
            rollback_to="",
            metadata={"rollback_from": str(current.version)},
        )
        self._current[env] = rollback_release
        self._history.append(rollback_release)
        self._save()
        return rollback_release

    def health_check(self, env: str = "production") -> dict:
        """Run health check on the deployed environment."""
        release = self._current.get(env)
        return {
            "env": env,
            "version": str(release.version) if release else "none",
            "status": release.status.value if release else "unknown",
            "traffic_pct": release.traffic_pct if release else 0,
            "deployed_at": release.deployed_at if release else "",
            "healthy": release is not None and release.status in (
                ReleaseStatus.ACTIVE, ReleaseStatus.ROLLING_OUT,
            ),
        }

    def deploy_status(self) -> dict:
        """Return deployment status across all environments."""
        return {
            env: {
                "version": str(r.version),
                "status": r.status.value,
                "traffic_pct": r.traffic_pct,
                "deployed_at": r.deployed_at,
            }
            for env, r in self._current.items()
        }

    def version_history(self, env: str | None = None, limit: int = 20) -> list[dict]:
        """Return version history, optionally filtered by env."""
        releases = self._history
        if env:
            releases = [r for r in releases if r.env == env]
        return [
            {
                "version": str(r.version),
                "env": r.env,
                "status": r.status.value,
                "deployed_at": r.deployed_at,
                "commit": r.commit_hash[:8] if r.commit_hash else "",
            }
            for r in releases[-limit:]
        ]
