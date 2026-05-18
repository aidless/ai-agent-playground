"""Kill old server + restart fresh on port 8000"""
import subprocess
import time
import sys

# 1. Kill
print("Killing old Python servers...")
r = subprocess.run('netstat -ano | findstr ":8000"', shell=True, capture_output=True, text=True)
pids = set()
for line in r.stdout.split('\n'):
    parts = line.strip().split()
    if len(parts) >= 5:
        pids.add(parts[-1])
for pid in pids:
    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
    print(f"  Killed {pid}")

time.sleep(2)

# 2. Start fresh
print("\nStarting server on http://localhost:8000 ...")
print("(Keep this window open — server stays alive)")
sys.stdout.flush()

subprocess.run([
    "uv", "run", "uvicorn", "agent.server:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--log-level", "warning"
])
