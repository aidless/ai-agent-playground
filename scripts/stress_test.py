"""50-concurrent stress test for P1 latency/reliability validation.

Usage: uv run python scripts/stress_test.py
"""

import asyncio
import statistics
import time
import httpx

BASE = "http://127.0.0.1:8000"
CONCURRENT = 50
TOTAL_REQUESTS = 500


async def worker(client: httpx.AsyncClient, path: str, results: list):
    start = time.perf_counter()
    try:
        r = await client.get(f"{BASE}{path}", timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        results.append({"path": path, "latency_ms": elapsed, "status": r.status_code})
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append({"path": path, "latency_ms": elapsed, "status": 0, "error": str(e)[:100]})


async def main():
    health_results = []
    clear_results = []
    mixed_results = []

    async with httpx.AsyncClient() as client:
        # Phase 1: Health endpoint saturation
        print(f"Phase 1: {CONCURRENT} concurrent to /health...")
        tasks = []
        for _ in range(TOTAL_REQUESTS):
            tasks.append(worker(client, "/health", health_results))
            if len(tasks) >= CONCURRENT:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

        health_ok = [r for r in health_results if r["status"] == 200]
        health_lats = [r["latency_ms"] for r in health_ok]
        print(f"  /health: {len(health_ok)}/{len(health_results)} OK")
        if health_lats:
            print(f"    avg={statistics.mean(health_lats):.0f}ms p50={sorted(health_lats)[len(health_lats)//2]:.0f}ms p95={sorted(health_lats)[int(len(health_lats)*0.95)]:.0f}ms p99={sorted(health_lats)[int(len(health_lats)*0.99)]:.0f}ms")

        # Phase 2: Mixed load
        print(f"\nPhase 2: {CONCURRENT} concurrent mixed endpoints...")
        endpoints = ["/health", "/clear", "/slo/report", "/memory/status", "/identity/list", "/tenancy/list", "/deploy/status", "/alerts/status", "/sandbox/audit", "/ciso/pending"]
        for _ in range(TOTAL_REQUESTS):
            ep = endpoints[_ % len(endpoints)]
            tasks.append(worker(client, ep, mixed_results))
            if len(tasks) >= CONCURRENT:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

        mixed_ok = [r for r in mixed_results if r["status"] == 200]
        mixed_lats = [r["latency_ms"] for r in mixed_ok]
        print(f"  Mixed: {len(mixed_ok)}/{len(mixed_results)} OK")
        if mixed_lats:
            print(f"    avg={statistics.mean(mixed_lats):.0f}ms p50={sorted(mixed_lats)[len(mixed_lats)//2]:.0f}ms p95={sorted(mixed_lats)[int(len(mixed_lats)*0.95)]:.0f}ms p99={sorted(mixed_lats)[int(len(mixed_lats)*0.99)]:.0f}ms")

    # Summary
    all_results = health_results + mixed_results
    all_ok = [r for r in all_results if r["status"] == 200]
    all_lats = [r["latency_ms"] for r in all_ok]
    print(f"\n{'='*55}")
    print(f"  STRESS TEST RESULTS ({CONCURRENT} concurrent)")
    print(f"{'='*55}")
    print(f"  Total: {len(all_results)} | OK: {len(all_ok)} ({len(all_ok)/len(all_results)*100:.0f}%)")
    if all_lats:
        sl = sorted(all_lats)
        print(f"  Avg:   {statistics.mean(all_lats):.0f}ms")
        print(f"  P50:   {sl[len(sl)//2]:.0f}ms")
        print(f"  P95:   {sl[int(len(sl)*0.95)]:.0f}ms")
        print(f"  P99:   {sl[int(len(sl)*0.99)]:.0f}ms")
        print(f"  Min:   {min(sl):.0f}ms  Max: {max(sl):.0f}ms")
        jitter = statistics.stdev(all_lats) / statistics.mean(all_lats) * 100 if len(all_lats) > 1 else 0
        # Per-category jitter (P1 requirement should be per-type, not mixed)
        by_label = {}
        for r in all_ok:
            lbl = "health" if "health" in r["path"] else "mixed"
            by_label.setdefault(lbl, []).append(r["latency_ms"])
        cat_jitters = {}
        for lbl, lats in by_label.items():
            if len(lats) > 2:
                cat_jitters[lbl] = statistics.stdev(lats) / statistics.mean(lats) * 100

        print(f"  Jitter (stddev/avg): {jitter:.1f}% [overall]")
        for lbl, j in cat_jitters.items():
            print(f"  Jitter ({lbl}): {j:.1f}% {'PASS' if j <= 30 else 'FAIL'}")
        print(f"  P1 target: p95<=3000ms {'PASS' if sl[int(len(sl)*0.95)] <= 3000 else 'FAIL'}")
        print(f"  P1 target: jitter<=30% {'PASS' if jitter <= 30 else 'FAIL (mixed types: ' + ' '.join(f'{l}={j:.0f}%' for l,j in cat_jitters.items()) + ')'}")
        print(f"  P2 target: p99<=5000ms {'PASS' if sl[int(len(sl)*0.99)] <= 5000 else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
