from pathlib import Path

import fitz  # pymupdf


def read_file(file_path: str) -> list[tuple[int, str]]:
    """Read a PDF or text file and return list of (page_number, text) tuples."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    elif suffix in (".txt", ".md", ".text"):
        text = path.read_text(encoding="utf-8")
        return [(1, text)]
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append((i + 1, text))
    return pages
