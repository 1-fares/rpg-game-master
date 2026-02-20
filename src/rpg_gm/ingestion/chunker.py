from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str
    page: int | None = None


def chunk_text(
    source: str | list[tuple[int, str]],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text into overlapping chunks using paragraph-aware boundaries.

    Splits on paragraph breaks (\\n\\n) first, then sentence boundaries.
    """
    if isinstance(source, str):
        pages = [(None, source)]
    else:
        pages = source

    # Build full text with page tracking
    full_text = ""
    page_breaks: list[tuple[int, int | None]] = []  # (char_offset, page_num)
    for page_num, text in pages:
        page_breaks.append((len(full_text), page_num))
        full_text += text

    if not full_text.strip():
        return []

    def get_page_at(offset: int) -> int | None:
        result = None
        for break_offset, page_num in page_breaks:
            if break_offset <= offset:
                result = page_num
            else:
                break
        return result

    # Split into paragraphs
    paragraphs = []
    for para in full_text.split("\n\n"):
        stripped = para.strip()
        if stripped:
            paragraphs.append(stripped)

    chunks = []
    current_parts: list[str] = []
    current_length = 0
    current_offset = 0  # Track position in full_text for page attribution
    idx = 0

    for para in paragraphs:
        para_len = len(para)

        # If adding this paragraph exceeds chunk_size and we have content, finalize
        if current_length + para_len > chunk_size and current_parts:
            chunk_text_str = "\n\n".join(current_parts)
            mid_offset = current_offset + len(chunk_text_str) // 2
            page = get_page_at(mid_offset)
            chunks.append(Chunk(index=idx, text=chunk_text_str, page=page))
            idx += 1

            # Keep last part(s) for overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part)
                else:
                    break

            current_offset += current_length - overlap_len
            current_parts = overlap_parts
            current_length = overlap_len

        current_parts.append(para)
        current_length += para_len

    # Don't forget the last chunk
    if current_parts:
        chunk_text_str = "\n\n".join(current_parts)
        mid_offset = current_offset + len(chunk_text_str) // 2
        page = get_page_at(mid_offset)
        chunks.append(Chunk(index=idx, text=chunk_text_str, page=page))

    return chunks
