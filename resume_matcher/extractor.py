"""Extract text from resume files (PDF, DOCX, TXT)."""

from pathlib import Path


def extract_resume_text(file_path: str) -> str:
    """Extract text from a resume file. Supports PDF, TXT, DOCX."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text.strip())
        return "\n\n".join(parts)

    elif ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    elif ext == ".docx":
        raise ValueError(
            "DOCX support requires python-docx. "
            "Convert to PDF or TXT, or install with: uv add python-docx"
        )

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")
