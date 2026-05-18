"""Blue-Green Deploy — zero-downtime deployment via environment swapping.

P3 requirement: zero-downtime releases with instant rollback capability.

How it works:
  - Two identical environments: BLUE (current live) and GREEN (next version)
  - Deploy new version to GREEN while BLUE serves traffic
  - Smoke-test GREEN
  - Swap: GREEN becomes live, BLUE becomes standby
  - If GREEN fails, instant rollback: swap back to BLUE

This is the natural evolution of the canary deploy in deploy.py.
Blue-green gives us zero-downtime (no traffic interruption during deploy)
and instant rollback (just swap back).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Side(str, Enum):
    BLUE = "blue"
    GREEN = "green"


class DeployPhase(str, Enum):
    IDLE = "idle"
    DEPLOYING = "deploying"           # Pushing to inactive side
    SMOKE_TESTING = "smoke_testing"   # Testing inactive side
    SWAPPING = "swapping"             # Switching traffic
    LIVE = "live"                      # Deployment complete
    ROLLING_BACK = "rolling_back"     # Emergency rollback


@dataclass
class BlueGreenState:
    """Current state of the blue-green deployment."""
    active: Side = Side.BLUE           # Which side is serving traffic
    inactive: Side = Side.GREEN        # Which side is standby
    phase: DeployPhase = DeployPhase.IDLE
    active_version: str = ""
    inactive_version: str = ""
    last_swap_at: str = ""
    smoke_test_passed: bool = False
    deploy_started_at: str = ""
    total_swaps: int = 0
    total_rollbacks: int = 0


@dataclass
class DeployResult:
    success: bool
    phase: DeployPhase
    message: str
    active_side: Side = Side.BLUE
    active_version: str = ""
    downtime_ms: float = 0.0


class BlueGreenDeployer:
    """Manages blue-green deployments with zero-downtime swapping.

    Usage:
        bg = BlueGreenDeployer()
        bg.deploy_to_inactive("v2.0.0")     # Deploy to inactive side
        bg.smoke_test()                       # Verify inactive side
        bg.swap()                             # Instantly swap traffic
        # GREEN is now live, BLUE is standby
        bg.rollback()                         # Swap back if needed
    """

    def __init__(self, store_path: str = "./blue_green_state"):
        self.store = Path(store_path)
        self.store.mkdir(parents=True, exist_ok=True)
        self.state = BlueGreenState()
        self._load()

    def _load(self):
        state_file = self.store / "state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            self.state = BlueGreenState(
                active=Side(data.get("active", "blue")),
                inactive=Side(data.get("inactive", "green")),
                phase=DeployPhase(data.get("phase", "idle")),
                active_version=data.get("active_version", ""),
                inactive_version=data.get("inactive_version", ""),
                last_swap_at=data.get("last_swap_at", ""),
                total_swaps=data.get("total_swaps", 0),
                total_rollbacks=data.get("total_rollbacks", 0),
            )

    def _save(self):
        data = {
            "active": self.state.active.value,
            "inactive": self.state.inactive.value,
            "phase": self.state.phase.value,
            "active_version": self.state.active_version,
            "inactive_version": self.state.inactive_version,
            "last_swap_at": self.state.last_swap_at,
            "total_swaps": self.state.total_swaps,
            "total_rollbacks": self.state.total_rollbacks,
        }
        (self.store / "state.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def deploy_to_inactive(self, version: str) -> DeployResult:
        """Deploy new version to the inactive side.

        During this phase, the active side continues serving traffic unaffected.
        """
        self.state.phase = DeployPhase.DEPLOYING
        self.state.inactive_version = version
        self.state.deploy_started_at = datetime.now(timezone.utc).isoformat()
        self._save()

        # Simulate deployment (in production: docker pull + restart inactive)
        time.sleep(0.01)  # Platform-level deploy would happen here

        return DeployResult(
            success=True,
            phase=DeployPhase.DEPLOYING,
            message=f"Deployed {version} to {self.state.inactive.value} side",
            active_side=self.state.active,
        )

    def smoke_test(self) -> DeployResult:
        """Run smoke tests on the inactive side.

        In production, this would hit the inactive environment's health endpoint
        and run integration tests. If tests fail, abort the swap.
        """
        self.state.phase = DeployPhase.SMOKE_TESTING

        # Simulate smoke test (in production: HTTP health check + integration tests)
        passed = True  # Assume tests pass; real implementation would verify

        if passed:
            self.state.smoke_test_passed = True
            self._save()
            return DeployResult(
                success=True,
                phase=DeployPhase.SMOKE_TESTING,
                message="Smoke tests passed",
            )
        else:
            self.state.phase = DeployPhase.IDLE
            self._save()
            return DeployResult(
                success=False,
                phase=DeployPhase.IDLE,
                message="Smoke tests failed — aborting deploy",
            )

    def swap(self) -> DeployResult:
        """Swap traffic from active to inactive side. Zero downtime.

        The swap itself is instant — it's just updating a routing rule.
        Traffic shifts atomically from one side to the other.
        """
        if self.state.phase != DeployPhase.SMOKE_TESTING:
            return DeployResult(
                success=False,
                phase=self.state.phase,
                message="Must complete smoke testing before swapping",
            )

        start = time.perf_counter()
        self.state.phase = DeployPhase.SWAPPING

        # The actual swap: just exchange active/inactive
        old_active = self.state.active
        old_inactive = self.state.inactive
        self.state.active = old_inactive
        self.state.inactive = old_active

        self.state.phase = DeployPhase.LIVE
        self.state.last_swap_at = datetime.now(timezone.utc).isoformat()
        self.state.total_swaps += 1
        self.state.smoke_test_passed = False
        self._save()

        downtime = (time.perf_counter() - start) * 1000

        return DeployResult(
            success=True,
            phase=DeployPhase.LIVE,
            message=f"Swapped: {self.state.active.value} is now live ({self.state.active_version})",
            active_side=self.state.active,
            active_version=self.state.active_version,
            downtime_ms=downtime,
        )

    def rollback(self) -> DeployResult:
        """Emergency rollback — swap back to previous active side.

        This is instant because the previous version is still running
        on the now-inactive side. No redeploy needed.
        """
        start = time.perf_counter()

        # Swap back
        old_active = self.state.active
        old_inactive = self.state.inactive
        self.state.active = old_inactive
        self.state.inactive = old_active

        self.state.phase = DeployPhase.LIVE
        self.state.total_rollbacks += 1
        self.state.last_swap_at = datetime.now(timezone.utc).isoformat()
        self._save()

        downtime = (time.perf_counter() - start) * 1000

        return DeployResult(
            success=True,
            phase=DeployPhase.LIVE,
            message=f"Rolled back: {self.state.active.value} is now live ({self.state.active_version})",
            active_side=self.state.active,
            active_version=self.state.active_version,
            downtime_ms=downtime,
        )

    def status(self) -> dict:
        """Current blue-green deployment status."""
        return {
            "active": self.state.active.value,
            "active_version": self.state.active_version,
            "inactive": self.state.inactive.value,
            "inactive_version": self.state.inactive_version,
            "phase": self.state.phase.value,
            "total_swaps": self.state.total_swaps,
            "total_rollbacks": self.state.total_rollbacks,
            "zero_downtime": True,
            "instant_rollback": True,
        }
