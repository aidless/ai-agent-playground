"""
Docker Sandbox — safe code execution in isolated containers.

Every AI agent that can write & run code needs a sandbox.
Without it, the agent can `rm -rf /`, access your files, or mine crypto.
This module wraps Docker to create throwaway containers with hard limits:
  - No network access (--network none)
  - Memory cap (--memory=256m)
  - CPU cap (--cpus=1)
  - Fork-bomb protection (--pids-limit=50)
  - Read-only root filesystem (--read-only)
  - Auto-cleanup after execution (--rm)
  - Hard timeout (30s wall clock)

Usage:
    sandbox = DockerSandbox()
    result = sandbox.execute("print(sum(range(100)))")
    # → "4950"
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path


class SandboxError(RuntimeError):
    """Raised when sandbox execution fails."""


class DockerSandbox:
    """Execute code inside an isolated Docker container."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory: str = "256m",
        cpu_limit: str = "1.0",
        pid_limit: int = 50,
    ):
        self.image = image
        self.timeout = timeout
        self.memory = memory
        self.cpu_limit = cpu_limit
        self.pid_limit = pid_limit

    # ----- Public API -----

    @staticmethod
    def is_available() -> bool:
        """Check if Docker is installed and the daemon is reachable."""
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def execute(self, code: str, language: str = "python") -> str:
        """Run code inside a disposable Docker container.

        Returns stdout+stderr combined. Raises SandboxError on failure.
        """
        if language != "python":
            raise SandboxError(f"Unsupported language: {language}. Only 'python' is supported.")

        if not self.is_available():
            raise SandboxError(
                "Docker is not available. Install Docker Desktop from docker.com"
            )

        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp_dir:
            script_path = Path(tmp_dir) / "script.py"
            script_path.write_text(code, encoding="utf-8")

            cmd = self._build_command(str(tmp_dir))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10,
                )
            except subprocess.TimeoutExpired:
                raise SandboxError(
                    f"Sandbox execution timed out after {self.timeout}s. "
                    "The code may have an infinite loop or be too slow."
                )

            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr

            return output.strip() if output.strip() else "(no output)"

    # ----- Internals -----

    def _build_command(self, mount_dir: str) -> list[str]:
        """Build the docker run command with all security flags.

        Flag-by-flag explanation:
          --rm              Delete container after exit
          --network none    No network access (can't download malware)
          --memory=N        Hard memory cap (container gets OOM-killed)
          --memory-swap=N   Same as memory — no swap, can't bypass limit
          --pids-limit=N    Max PIDs in container (prevents fork bombs)
          --read-only       Root FS is read-only
          --tmpfs /tmp:rw,noexec  /tmp is writable but code can't execute from it
          --cpus=N          Max CPU cores
          --stop-timeout=N  Seconds to wait before SIGKILL
          -v host:container Mount code read-only into container
        """
        return [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--pids-limit", str(self.pid_limit),
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec",
            "--cpus", self.cpu_limit,
            "--stop-timeout", "5",
            "-v", f"{mount_dir}:/code:ro",
            self.image,
            "timeout", str(self.timeout), "python", "/code/script.py",
        ]


# Module-level singleton
_sandbox: DockerSandbox | None = None


def get_sandbox() -> DockerSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = DockerSandbox()
    return _sandbox
