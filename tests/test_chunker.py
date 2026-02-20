from rpg_gm.ingestion.chunker import Chunk, chunk_text


def test_chunk_short_text():
    """Text shorter than chunk_size produces one chunk."""
    result = chunk_text("Short paragraph.")
    assert len(result) == 1
    assert result[0].text == "Short paragraph."
    assert result[0].index == 0


def test_chunk_splits_on_paragraphs():
    """Long text with paragraph breaks splits at paragraph boundaries."""
    para = "A" * 800
    text = f"{para}\n\n{para}\n\n{para}"
    result = chunk_text(text, chunk_size=1200, overlap=150)
    assert len(result) > 1
    # Each chunk text should not contain partial paragraphs
    for chunk in result:
        # Should not start/end mid-word since we split on \n\n
        assert chunk.text.strip() == chunk.text


def test_chunk_overlap():
    """Adjacent chunks share some content."""
    # Use short paragraphs so they fit within the overlap budget
    paras = [f"Paragraph {i}. " + "x" * 80 for i in range(20)]
    text = "\n\n".join(paras)
    result = chunk_text(text, chunk_size=500, overlap=150)
    assert len(result) >= 2
    # Check that consecutive chunks share text
    for i in range(len(result) - 1):
        current_text = result[i].text
        next_text = result[i + 1].text
        # The overlap means some paragraph from the end of current
        # should appear at the start of next
        current_paras = set(current_text.split("\n\n"))
        next_paras = set(next_text.split("\n\n"))
        shared = current_paras & next_paras
        assert len(shared) > 0, "Adjacent chunks should share at least one paragraph"


def test_chunk_page_numbers():
    """Chunks from multi-page input have correct page numbers."""
    # Each page has multiple paragraphs separated by \n\n so chunking can split
    page1_paras = "\n\n".join([f"Page one paragraph {i}. " + "a" * 100 for i in range(8)])
    page2_paras = "\n\n".join([f"Page two paragraph {i}. " + "b" * 100 for i in range(8)])
    pages = [
        (1, page1_paras),
        (2, page2_paras),
    ]
    result = chunk_text(pages, chunk_size=500, overlap=100)
    assert len(result) >= 2
    # First chunk should be from page 1
    assert result[0].page == 1
    # Last chunk should be from page 2
    assert result[-1].page == 2


def test_chunk_empty_text():
    """Empty input returns empty list."""
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []
    assert chunk_text([]) == []


def test_chunk_dataclass():
    """Chunk fields work correctly."""
    c = Chunk(index=3, text="hello", page=7)
    assert c.index == 3
    assert c.text == "hello"
    assert c.page == 7

    c2 = Chunk(index=0, text="no page")
    assert c2.page is None
