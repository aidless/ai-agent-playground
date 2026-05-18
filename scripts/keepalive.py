"""Server keepalive watchdog — monitors health and restarts on crash.

Usage: uv run python scripts/keepalive.py [--interval 10]
"""

import subprocess
import time
import sys
import httpx

INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
BASE = "http://127.0.0.1:8000"
MAX_RESTARTS = 5

print(f"Watchdog started (interval={INTERVAL}s)")

restarts = 0
while restarts < MAX_RESTARTS:
    # Check if server is alive
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        if r.status_code == 200:
            time.sleep(INTERVAL)
            continue
    except Exception:
        pass

    # Server is down — restart
    restarts += 1
    print(f"[{time.strftime('%H:%M:%S')}] Server down. Restarting (#{restarts})...")

    # Kill stale processes
    r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in r.stdout.split("\n"):
        if ":8000" in line and "LISTENING" in line:
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)

    # Start new server
    subprocess.Popen(
        ["uv", "run", "python", "agent/server.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Wait for it to come up
    for _ in range(20):
        time.sleep(1)
        try:
            r = httpx.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                print(f"[{time.strftime('%H:%M:%S')}] Server recovered")
                break
        except Exception:
            continue

print(f"Max restarts ({MAX_RESTARTS}) reached. Watchdog exiting.")
