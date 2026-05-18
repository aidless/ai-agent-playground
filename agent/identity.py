"""Identity & Access Management — per-agent identity + permission model.

Every agent and tool invocation carries a unique identity. Permissions follow
the principle of least privilege with short-lived credentials.

Identity model:
  - Principal: who (user / agent / service)
  - Role: what permissions the principal has
  - Credential: short-lived proof of identity (API key / JWT / session token)
  - Session: a traceable interaction window
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Permission(str, Enum):
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    EXECUTE_CODE = "execute_code"
    NETWORK_OUT = "network_out"
    NETWORK_IN = "network_in"
    MANAGE_TOOLS = "manage_tools"
    MANAGE_AGENTS = "manage_agents"
    ADMIN = "admin"


class Role(str, Enum):
    VIEWER = "viewer"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    ADMIN = "admin"


ROLE_PERMISSIONS = {
    Role.VIEWER: [Permission.READ_FILES],
    Role.DEVELOPER: [
        Permission.READ_FILES,
        Permission.WRITE_FILES,
        Permission.EXECUTE_CODE,
        Permission.NETWORK_OUT,
    ],
    Role.OPERATOR: [
        Permission.READ_FILES,
        Permission.WRITE_FILES,
        Permission.EXECUTE_CODE,
        Permission.NETWORK_OUT,
        Permission.NETWORK_IN,
        Permission.MANAGE_TOOLS,
    ],
    Role.ADMIN: [p for p in Permission],
}


@dataclass
class Identity:
    """Unique identity for an agent or user."""

    id: str
    name: str
    role: Role = Role.DEVELOPER
    api_key_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = ""
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def permissions(self) -> list[Permission]:
        return ROLE_PERMISSIONS.get(self.role, [])

    def has_permission(self, perm: Permission) -> bool:
        return perm in self.permissions

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "active": self.active,
        }


@dataclass
class SessionToken:
    """Short-lived session credential."""

    token: str
    identity_id: str
    issued_at: float
    expires_at: float
    scopes: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def ttl_seconds(self) -> float:
        return max(0, self.expires_at - time.time())


class IdentityManager:
    """Manages identities, roles, and session tokens.

    Usage:
        im = IdentityManager()
        agent_id = im.register_identity("code-reviewer", Role.DEVELOPER)
        token = im.issue_token(agent_id, ttl_minutes=60)
        identity = im.validate_token(token)
    """

    def __init__(self, store_path: str = "./sandbox_workspace/identities"):
        self.store_dir = Path(store_path)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.store_file = self.store_dir / "identities.json"
        self._identities: dict[str, Identity] = {}
        self._sessions: dict[str, SessionToken] = {}
        self._signing_key = os.environ.get("IDENTITY_SIGNING_KEY", secrets.token_hex(32))
        self._load()

    def _load(self):
        """Load identities from persistent store."""
        if self.store_file.exists():
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
            for id_data in data.get("identities", []):
                ident = Identity(
                    id=id_data["id"],
                    name=id_data["name"],
                    role=Role(id_data["role"]),
                    api_key_hash=id_data.get("api_key_hash", ""),
                    created_at=id_data.get("created_at", ""),
                    last_seen=id_data.get("last_seen", ""),
                    active=id_data.get("active", True),
                )
                self._identities[ident.id] = ident

    def _save(self):
        """Persist identities to disk."""
        data = {
            "identities": [
                {
                    "id": i.id,
                    "name": i.name,
                    "role": i.role.value,
                    "api_key_hash": i.api_key_hash,
                    "created_at": i.created_at,
                    "last_seen": i.last_seen,
                    "active": i.active,
                }
                for i in self._identities.values()
            ]
        }
        self.store_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def register_identity(self, name: str, role: Role, metadata: dict = None) -> str:
        """Register a new agent/user identity. Returns identity ID."""
        identity_id = f"id-{secrets.token_hex(6)}"
        ident = Identity(
            id=identity_id,
            name=name,
            role=role,
            metadata=metadata or {},
        )
        self._identities[identity_id] = ident
        self._save()
        return identity_id

    def get_identity(self, identity_id: str) -> Identity | None:
        return self._identities.get(identity_id)

    def issue_token(self, identity_id: str, ttl_minutes: int = 15) -> str:
        """Issue a short-lived session token (default 15min for zero-trust)."""
        ident = self.get_identity(identity_id)
        if not ident or not ident.active:
            raise ValueError(f"Identity {identity_id} not found or inactive")

        now = time.time()
        token_raw = f"{identity_id}:{now}:{secrets.token_hex(8)}"
        signature = hmac.new(
            self._signing_key.encode(),
            token_raw.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]
        token = f"sk-sess-{token_raw.replace(':','.')}.{signature}"

        self._sessions[token] = SessionToken(
            token=token,
            identity_id=identity_id,
            issued_at=now,
            expires_at=now + ttl_minutes * 60,
            scopes=[p.value for p in ident.permissions],
        )

        ident.last_seen = datetime.now(timezone.utc).isoformat()
        self._save()
        return token

    def validate_token(self, token: str) -> Identity | None:
        """Validate a session token and return the associated identity."""
        session = self._sessions.get(token)
        if not session:
            return None
        if session.is_expired:
            del self._sessions[token]
            return None
        return self.get_identity(session.identity_id)

    def revoke_token(self, token: str):
        """Revoke a session token."""
        self._sessions.pop(token, None)

    def cleanup_expired(self):
        """Remove expired sessions."""
        expired = [t for t, s in self._sessions.items() if s.is_expired]
        for t in expired:
            del self._sessions[t]

    def has_permission(self, identity_id: str, permission: Permission) -> bool:
        """Check if an identity has a specific permission."""
        ident = self.get_identity(identity_id)
        if not ident or not ident.active:
            return False
        return ident.has_permission(permission)

    def list_identities(self) -> list[dict]:
        """List all registered identities."""
        return [i.to_dict() for i in self._identities.values()]

    def disable_identity(self, identity_id: str):
        """Deactivate an identity."""
        ident = self.get_identity(identity_id)
        if ident:
            ident.active = False
            self._save()

    def enable_identity(self, identity_id: str):
        """Reactivate an identity."""
        ident = self.get_identity(identity_id)
        if ident:
            ident.active = True
            self._save()
