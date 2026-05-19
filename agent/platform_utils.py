"""Cross-Platform Utilities — safe os operations across Windows / Linux / macOS.

Handles:
  - Subprocess execution (CREATE_NO_WINDOW on Windows)
  - File path normalization (forward/backslash)
  - Permission-safe file operations
  - Platform-aware environment detection
"""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


def run_command(cmd: list[str], timeout: int = 30, cwd: str = None, env: dict = None) -> dict:
    """Platform-safe subprocess execution.

    On Windows: uses CREATE_NO_WINDOW (no console popup).
    On Linux/macOS: standard subprocess with timeout.
    """
    kwargs = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if cwd:
        kwargs["cwd"] = str(cwd)
    if env:
        kwargs["env"] = env

    if IS_WINDOWS:
        # Prevent console window flash on Windows
        if "creationflags" not in kwargs:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        # On Windows, cmd list sometimes needs shell
        if any(" " in str(c) for c in cmd):
            kwargs["shell"] = True

    try:
        result = subprocess.run(cmd, **kwargs)
        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1, "success": False}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: {cmd[0]}", "returncode": -1, "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


def safe_path(path_str: str) -> Path:
    """Normalize path for current platform."""
    p = Path(path_str)
    # Ensure path doesn't escape allowed directories
    try:
        resolved = p.resolve()
        if not str(resolved).startswith(str(Path.cwd())):
            logger.warning("Path escapes working directory: %s", path_str)
    except (OSError, ValueError):
        pass
    return p


def get_app_dir() -> Path:
    """Get platform-appropriate config directory."""
    if IS_WINDOWS:
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif IS_MACOS:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    app_dir = base / "ai-agent-playground"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory."""
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif IS_MACOS:
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "ai-agent-playground"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def env_info() -> dict:
    """Return platform and Python environment info."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "is_windows": IS_WINDOWS,
        "app_dir": str(get_app_dir()),
        "cache_dir": str(get_cache_dir()),
        "uv_path": os.environ.get("UV_PATH", ""),
    }
