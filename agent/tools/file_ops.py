"""文件操作工具：读/写/列文件（限项目目录内）"""

import os
from pathlib import Path

TOOLS = []

# 安全限制：只允许操作项目目录内的文件
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIRS = [PROJECT_ROOT]
BLOCKED_PATTERNS = (
    ".env",
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".pytest_cache",
    ".hermes_test_result.txt",
    ".fix_tokenizers.py",
    ".run_test.py",
)
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


def _resolve_path(path_str: str) -> Path | None:
    """解析并校验路径是否在允许范围内"""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()

    # 必须在项目根目录内
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    # 不能是拦截名单
    for blocked in BLOCKED_PATTERNS:
        if blocked in p.parts:
            return None

    return p


# ── 读文件 ──

READ_DEF = {
    "name": "read_file",
    "description": "读取文件内容（限项目目录内，最大 1MB）",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径（相对或绝对）"},
        },
        "required": ["path"],
    },
}


def read_file(path: str) -> str:
    resolved = _resolve_path(path)
    if resolved is None:
        return "错误：路径不在允许范围内或被禁止"
    if not resolved.exists():
        return f"错误：文件不存在 ({resolved})"
    if not resolved.is_file():
        return f"错误：不是文件 ({resolved})"
    if resolved.stat().st_size > MAX_FILE_SIZE:
        return f"错误：文件超过 1MB 限制 ({resolved.stat().st_size / 1024:.0f} KB)"

    try:
        content = resolved.read_text(encoding="utf-8")
        return content
    except UnicodeDecodeError:
        return f"[二进制文件，{resolved.stat().st_size} bytes]"
    except Exception as e:
        return f"读取失败: {e}"


TOOLS.append((READ_DEF, read_file))


# ── 写文件 ──

WRITE_DEF = {
    "name": "write_file",
    "description": "写入文件（覆盖已有内容，限项目目录内）。会先让用户确认。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径（相对或绝对）"},
            "content": {"type": "string", "description": "文件内容"},
        },
        "required": ["path", "content"],
    },
}


def write_file(path: str, content: str) -> str:
    resolved = _resolve_path(path)
    if resolved is None:
        return "错误：路径不在允许范围内或被禁止"

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字符到 {resolved.relative_to(PROJECT_ROOT)}"
    except Exception as e:
        return f"写入失败: {e}"


TOOLS.append((WRITE_DEF, write_file))


# ── 列目录 ──

LIST_DEF = {
    "name": "list_files",
    "description": "列出目录下的文件和子目录（限项目目录内）",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径（默认项目根目录）"},
            "depth": {"type": "integer", "description": "递归深度（0=当前目录，1=子目录一层，默认 0）"},
        },
        "required": [],
    },
}


def list_files(path: str = ".", depth: int = 0) -> str:
    resolved = _resolve_path(path)
    if resolved is None:
        return "错误：路径不在允许范围内"
    if not resolved.exists():
        return f"错误：目录不存在 ({resolved})"
    if not resolved.is_dir():
        return f"错误：不是目录 ({resolved})"

    depth = max(0, min(depth, 2))  # 限制最大深度 2

    lines = []

    def _walk(current: Path, cur_depth: int):
        if cur_depth > depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            rel = entry.relative_to(PROJECT_ROOT)
            if entry.is_dir():
                lines.append(f"{'  ' * cur_depth}{'└── ' if cur_depth < depth else ''}{entry.name}/")
                _walk(entry, cur_depth + 1)
            else:
                size = entry.stat().st_size
                size_str = f"{size}B" if size < 1024 else f"{size / 1024:.0f}KB"
                lines.append(f"{'  ' * cur_depth}{entry.name} ({size_str})")

    _walk(resolved, 0)
    return "\n".join(lines) if lines else "(空目录)"


TOOLS.append((LIST_DEF, list_files))
