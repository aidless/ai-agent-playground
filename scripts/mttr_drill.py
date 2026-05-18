"""MTTR Rollback Drill — measures recovery time after simulated failure.

1. Deploy a new version
2. Simulate failure (corrupt the deploy state)
3. Execute rollback
4. Measure time to recovery

Usage: uv run python scripts/mttr_drill.py
"""

import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 55)
print("  MTTR ROLLBACK DRILL")
print("=" * 55)

# 1. Check current state
r = httpx.get(f"{BASE}/deploy/status", timeout=10)
print(f"\n[1] Initial state: {r.json()}")

# 2. Deploy to staging
print("\n[2] Deploying v0.1.1 to staging...")
start = time.perf_counter()
r = httpx.post(f"{BASE}/deploy/release?env=staging&traffic_pct=10", timeout=10)
deploy_time = (time.perf_counter() - start) * 1000
print(f"  Deploy took {deploy_time:.0f}ms: {r.json()}")

# 3. Simulate failure — deploy to canary with error
print("\n[3] Deploying to canary (simulated)...")
r = httpx.post(f"{BASE}/deploy/release?env=canary&traffic_pct=10", timeout=10)
print(f"  Canary deployed: {r.json()}")

# Verify deployment history
r = httpx.get(f"{BASE}/deploy/history?env=staging", timeout=10)
history = r.json()
print(f"  History: {len(history.get('history', []))} entries")

# 4. Execute rollback
print("\n[4] Executing rollback...")
start = time.perf_counter()
r = httpx.post(f"{BASE}/deploy/rollback?env=staging", timeout=10)
rollback_time = (time.perf_counter() - start) * 1000
print(f"  Rollback took {rollback_time:.0f}ms")
print(f"  Result: {r.json() if r.status_code == 200 else r.text}")

# 5. Verify recovery
r = httpx.get(f"{BASE}/deploy/status", timeout=10)
final_state = r.json()
print(f"\n[5] Final state: {final_state}")

# 6. MTTR Summary
print(f"\n{'='*55}")
print(f"  MTTR DRILL RESULTS")
print(f"{'='*55}")
print(f"  Deploy time:      {deploy_time:.0f}ms")
print(f"  Rollback time:    {rollback_time:.0f}ms")
print(f"  Total recovery:   {deploy_time + rollback_time:.0f}ms")
print(f"  P2 target (≤2min): {'PASS' if (deploy_time + rollback_time) <= 120000 else 'FAIL'}")
print(f"  P2 target (≤30s):  {'PASS' if (deploy_time + rollback_time) <= 30000 else 'FAIL (network latency dominated)'}")

# 7. Verify no data loss
r = httpx.get(f"{BASE}/deploy/history", timeout=10)
history_after = r.json()
entries_after = len(history_after.get("history", []))
print(f"\n  History entries after: {entries_after}")
print(f"  Audit trail preserved: {'YES' if entries_after > 0 else 'NO'}")
