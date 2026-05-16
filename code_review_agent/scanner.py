"""Code scanner: walk a directory and collect code files for review."""

import os
from dataclasses import dataclass
from pathlib import Path

# File extensions we know how to review
CODE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React JSX",
    ".tsx": "React TSX",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cpp": "C++",
    ".css": "CSS",
    ".html": "HTML",
    ".sql": "SQL",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules",
             ".idea", ".vscode", "dist", "build", "target", ".next"}

MAX_FILE_BYTES = 200_000  # ~200KB, avoid sending huge files to AI


@dataclass
class FileInfo:
    abs_path: str
    rel_path: str
    language: str
    content: str
    lines: int


def scan_directory(root: str) -> list[FileInfo]:
    """Walk root, return code files ready for review. Binary/skipped dirs excluded."""
    files = []
    root_path = Path(root).resolve()

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            abs_path = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(abs_path)
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue

            files.append(FileInfo(
                abs_path=abs_path,
                rel_path=str(Path(abs_path).relative_to(root_path)),
                language=CODE_EXTENSIONS[ext],
                content=content,
                lines=content.count("\n") + 1,
            ))

    # Sort by language then rel_path for stable output
    files.sort(key=lambda f: (f.language, f.rel_path))
    return files
