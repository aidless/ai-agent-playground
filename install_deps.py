"""临时脚本：安装 pyproject.toml 中更新的依赖"""
import subprocess, sys

result = subprocess.run([sys.executable, "-m", "uv", "sync"], capture_output=True, text=True, cwd=r"C:\Users\Administrator\Desktop\ai-agent-playground")
if result.returncode != 0:
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
else:
    print(result.stdout or "uv sync completed successfully")
