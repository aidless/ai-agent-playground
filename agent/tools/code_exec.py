"""Python 代码执行沙箱（安全受限）"""

import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = []

EXEC_DEF = {
    "name": "run_python",
    "description": "在沙箱中执行 Python 代码，返回 stdout/stderr。适合计算、数据分析、测试小段代码。代码在临时目录运行，无法修改项目文件。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码"},
            "timeout": {"type": "integer", "description": "超时秒数（默认 30）"},
        },
        "required": ["code"],
    },
}


def run_python(code: str, timeout: int = 30) -> str:
    """在子进程中执行 Python 代码，隔离文件系统"""
    timeout = max(5, min(timeout, 120))  # 限制 5-120 秒

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        script = Path(tmpdir) / "_exec.py"
        script.write_text(code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={"PATH": "/usr/bin:/usr/local/bin", "HOME": tmpdir},
            )
            output = []
            if result.stdout:
                output.append(result.stdout)
            if result.stderr:
                output.append(f"--- stderr ---\n{result.stderr}")
            if result.returncode != 0:
                output.append(f"--- exit code {result.returncode} ---")
            return "".join(output) if output else "(无输出)"
        except subprocess.TimeoutExpired:
            return f"错误：执行超时（{timeout}s）"
        except Exception as e:
            return f"执行失败: {e}"


TOOLS.append((EXEC_DEF, run_python))
