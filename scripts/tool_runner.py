"""工具调用容错包装 — 解决"工具链脆" (#3)

对每个工具调用自动：
    1. 重试（指数退避）
    2. 超时保护
    3. 回退策略
    4. 错误日志

用法:
    from scripts.tool_runner import run_with_retry
    result = run_with_retry(lambda: some_tool_call(), name="web_search")
"""

import json
import logging
import time
import traceback
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# 回退策略映射
FALLBACKS: dict[str, Callable] = {
    "web_search": lambda: json.dumps({"ok": False, "data": [{"error": "All search backends failed — network may be restricted"}]}),
    "read_file": lambda: "(file read failed after retries)",
    "bash": lambda: "(command execution failed — output not available)",
}


def run_with_retry(
    func: Callable,
    name: str = "unknown",
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 120.0,
) -> tuple[Any, bool]:
    """带重试和回退的工具调用包装

    Returns:
        (result, success)
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = func()
            if result is not None:
                return result, True
            # None 也算成功（比如 bash 无输出）
            return result, True
        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)
            logger.warning("[%s] attempt %d/%d failed: %s — retrying in %.1fs", name, attempt + 1, max_retries, e, delay)
            if attempt < max_retries - 1:
                time.sleep(delay)

    # 所有重试耗尽 → 回退
    fallback = FALLBACKS.get(name)
    if fallback:
        try:
            logger.info("[%s] using fallback after %d failures", name, max_retries)
            return fallback(), False
        except Exception:
            pass

    logger.error("[%s] all retries exhausted: %s\n%s", name, last_error, traceback.format_exc()[-500:])
    return f"Tool '{name}' failed after {max_retries} attempts: {last_error}", False


# ── 常用工具的高层封装 ──

def safe_bash(command: str, timeout: float = 120.0) -> tuple[str, bool]:
    """执行 bash 命令，自动包装 Git Bash 路径 + 超时 + 重试"""
    import subprocess

    full_cmd = f'"C:/Program Files/Git/bin/bash.exe" -c "{command}"'

    def _run():
        r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or r.stderr.strip() or "(no output)"

    return run_with_retry(_run, name="bash", max_retries=2)


def safe_search(query: str, max_results: int = 5) -> tuple[str, bool]:
    """安全搜索，自动处理网络问题"""
    import subprocess

    def _run():
        r = subprocess.run(
            ["uv", "run", "python", "scripts/search_web.py", "search", query, "--max-results", str(max_results)],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip() or json.dumps({"ok": False, "data": [{"error": r.stderr[:200]}]})

    return run_with_retry(_run, name="web_search", max_retries=2)


def safe_pytest(test_path: str = "tests/", timeout: float = 120.0) -> tuple[str, bool]:
    """安全运行测试"""
    import subprocess

    def _run():
        r = subprocess.run(
            ["uv", "run", "python", "-m", "pytest", test_path, "-q", "--tb=line"],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() or r.stderr.strip()

    return run_with_retry(_run, name="pytest", max_retries=1)
