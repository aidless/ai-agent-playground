"""检查本地运行 Hermes 的环境"""
import subprocess, sys, os, shutil

def check(cmd, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"[{label}] {'OK' if r.returncode == 0 else 'FAIL'} | {r.stdout.strip()[:80] or r.stderr.strip()[:80]}")
    except FileNotFoundError:
        print(f"[{label}] NOT FOUND")
    except Exception as e:
        print(f"[{label}] ERROR: {e}")

print("=== Hermes 环境检查 ===")
check(["ollama", "--version"], "ollama")
check(["nvidia-smi"], "nvidia-smi")

# Check transformers in .venv
import importlib
for mod in ["torch", "transformers", "bitsandbytes", "vllm"]:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"[{mod}] OK v{ver}")
    except ImportError:
        print(f"[{mod}] NOT INSTALLED")

print(f"\nPython: {sys.version}")
print(f"Platform: {sys.platform}")
