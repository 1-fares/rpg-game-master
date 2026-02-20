import pytest
from pathlib import Path
from rpg_gm.ingestion.reader import read_file


def test_read_text_file(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world\nLine two", encoding="utf-8")
    result = read_file(str(f))
    assert result == [(1, "Hello world\nLine two")]


def test_read_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        read_file("/nonexistent/path/fake.txt")


def test_read_unsupported_extension(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_file(str(f))


def test_read_md_file(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("# Heading\nBody text", encoding="utf-8")
    result = read_file(str(f))
    assert result == [(1, "# Heading\nBody text")]
