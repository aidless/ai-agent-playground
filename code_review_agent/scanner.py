"""Code scanner: walk a directory and collect code files for review.

Now config-driven: CODE_EXTENSIONS, SKIP_DIRS, MAX_FILE_BYTES come from
CodeReviewConfig, not module-level constants.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from .config import CodeReviewConfig


@dataclass
class FileInfo:
    abs_path: str
    rel_path: str
    language: str
    content: str
    lines: int


class Scanner:
    """Scans a directory and returns reviewable files.

    Like a tokenizer: raw filesystem → structured FileInfo objects.
    """

    def __init__(self, config: CodeReviewConfig):
        self.config = config

    def scan(self, root: str) -> list[FileInfo]:
        """Walk root, return code files ready for review."""
        files = []
        root_path = Path(root).resolve()

        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in self.config.skip_dirs]

            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in self.config.code_extensions:
                    continue

                abs_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    continue
                if size > self.config.max_file_bytes:
                    continue

                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, OSError):
                    continue

                files.append(FileInfo(
                    abs_path=abs_path,
                    rel_path=str(Path(abs_path).relative_to(root_path)),
                    language=self.config.code_extensions[ext],
                    content=content,
                    lines=content.count("\n") + 1,
                ))

        files.sort(key=lambda f: (f.language, f.rel_path))
        return files
