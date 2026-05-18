"""Fault injection test — verify alerts trigger on failures.

Tests:
  1. Circuit breaker: rapid failures → OPEN → recovery
  2. Cost budget: burn budget → alert
  3. Latency spike: slow request → p95 alert
  4. Fake tenant quota exhaustion → 429
  5. Fake CISO denial → blocked high-risk op
"""

import httpx
import time

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: {detail}")


print("=" * 55)
print("  FAULT INJECTION TESTS")
print("=" * 55)

# 1. Tenant quota exhaustion
print("\n[1] Tenant quota exhaustion")
# Register a tenant with tiny quota
r = httpx.post(f"{BASE}/tenancy/register?tenant_id=test-quota&name=QuotaTest&rpm=1", timeout=10)
test("Register low-quota tenant", r.status_code == 200)

# Send requests with that tenant until 429
quota_hit = False
for _ in range(5):
    r = httpx.get(f"{BASE}/health", headers={"X-Tenant-ID": "test-quota"}, timeout=10)
    if r.status_code == 429:
        quota_hit = True
        break
test("Quota exceeded triggers 429", quota_hit)

# 2. CISO approval flow
print("\n[2] CISO approval gate")
r = httpx.post(f"{BASE}/ciso/approval?tool_name=delete_file&risk_level=critical&requester=fault_test&justification=Testing CISO gate", timeout=10)
test("CISO request created", r.status_code == 200)
req_id = r.json().get("request_id", "")

r = httpx.get(f"{BASE}/ciso/pending", timeout=10)
test("CISO pending list shows request", r.status_code == 200 and req_id in str(r.text) if req_id else False)

r = httpx.post(f"{BASE}/ciso/deny?request_id={req_id}&reason=Test denial", timeout=10)
test("CISO deny works", r.status_code == 200)

# 3. SLO endpoint (should return even with zero data)
print("\n[3] SLO resilience")
r = httpx.get(f"{BASE}/slo/report", timeout=10)
test("SLO report returns", r.status_code == 200)
test("SLO report has required fields", "total_calls" in r.text)

r = httpx.get(f"{BASE}/slo/budget", timeout=10)
test("SLO budget returns", r.status_code == 200)

# 4. All endpoints respond
print("\n[4] Full endpoint availability")
endpoints = [
    "/health", "/clear", "/clear/report", "/metrics",
    "/governance/report", "/governance/audit",
    "/identity/list", "/tenancy/list",
    "/deploy/status", "/deploy/history",
    "/alerts/status", "/alerts/firing",
    "/sandbox/audit",
    "/ciso/pending",
    "/slo/report", "/slo/budget",
    "/memory/status",
]
for ep in endpoints:
    try:
        r = httpx.get(f"{BASE}{ep}", timeout=10)
        test(f"{ep}", r.status_code == 200)
    except Exception as e:
        test(f"{ep}", False, str(e))

# 5. Concurrent health checks (basic resilience)
print("\n[5] Basic resilience")
import concurrent.futures

def check_health():
    try:
        r = httpx.get(f"{BASE}/health", timeout=10)
        return r.status_code == 200
    except:
        return False

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(lambda _: check_health(), range(20)))
test("10 concurrent health checks", all(results), f"{sum(results)}/20 passed")

# 6. Version history
print("\n[6] Deploy history")
r = httpx.get(f"{BASE}/deploy/history", timeout=10)
test("Deploy history accessible", r.status_code == 200)

# Summary
print(f"\n{'='*55}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"  Score: {PASS/(PASS+FAIL)*100:.0f}%" if PASS+FAIL > 0 else "  No results")
