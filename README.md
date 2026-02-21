# RPG Game Master

A terminal-based RPG game master that turns any book or PDF into an explorable, interactive world. Feed it a history book, and you can walk through ancient cities, talk to historical figures, and discover lore — all grounded in the source material.

No combat. The focus is exploration, discovery, and conversation.

## How It Works

**Two-phase architecture:**

1. **Ingestion** (`uv run rpg-gm ingest <file>`) — Reads a PDF or text file, chunks the text, embeds it into ChromaDB for retrieval, then uses Claude to extract world entities (locations, NPCs, events, factions, lore). You review each extraction before it's added.

2. **Play** (`uv run rpg-gm play <world>`) — Loads the world and drops you into the first location. Navigate between locations, talk to NPCs, examine details. Every interaction uses RAG retrieval + Claude narration, so responses stay grounded in the source material.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/1-fares/rpg-game-master.git
cd rpg-game-master
uv sync
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Ingest a book

```bash
uv run rpg-gm ingest path/to/book.pdf --name "Ancient Athens"
```

This runs the full pipeline: read, chunk, embed, extract. During extraction, you'll be prompted to accept or skip each entity Claude finds:

```
Location: The Agora
  A bustling marketplace at the heart of Athens...
> a (accept), s (skip), q (quit)
```

### Play

```bash
uv run rpg-gm play ancient-athens
```

List available worlds:

```bash
uv run rpg-gm play
```

### In-game commands

| Command | Action |
|---------|--------|
| `/look` | Describe current location in detail |
| `/talk <npc>` | Start conversation with an NPC |
| `/examine <thing>` | Examine something in the environment |
| `/map` | Show visited locations and connections |
| `/journal` | Show journal entries |
| `/discoveries` | Show discovery progress |
| `/end` | End current NPC conversation |
| `/help` | Show all commands |
| `/quit` | Exit the game |

Free text input works too — just type what you want to do and Claude will narrate the result.

## Project Structure

```
src/rpg_gm/
  cli.py              # Entry points: ingest, play
  ingestion/
    reader.py          # PDF/text file reading (PyMuPDF)
    chunker.py         # Paragraph-aware text chunking
    embedder.py        # ChromaDB embedding + retrieval
    extractor.py       # Claude entity extraction (tool_use)
  world/
    models.py          # Pydantic models (Location, NPC, Event, ...)
    loader.py          # World JSON persistence
  game/
    engine.py          # Main game loop
    commands.py        # Command parsing + fuzzy name matching
    narrator.py        # Claude narration with RAG + streaming
    state.py           # SQLite game state
  ui/
    display.py         # Rich terminal output
```

## Technical Details

- **Chunking**: Paragraph-aware, 1200-char chunks with 150-char overlap. Preserves sentence boundaries and tracks page numbers.
- **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers. Runs on CPU, ~80MB model.
- **Entity extraction**: Claude with `tool_use` for structured output. Processes chunks in batches of 10.
- **Narration**: Claude with tool_use for state changes (movement, discoveries). Streaming responses with retry and exponential backoff.
- **State**: SQLite for game state (location, journal, discoveries, conversation history).
- **RAG**: Top-5 ChromaDB retrieval per interaction. Query varies by command type.

## Running Tests

```bash
uv run pytest
```

68 tests covering models, persistence, chunking, embedding, extraction, commands, state, and integration.

## Dependencies

- [anthropic](https://github.com/anthropics/anthropic-sdk-python) — Claude API
- [chromadb](https://github.com/chroma-core/chroma) — Vector store
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — Local embeddings
- [pymupdf](https://github.com/pymupdf/PyMuPDF) — PDF reading
- [rich](https://github.com/Textualize/rich) — Terminal UI
- [pydantic](https://github.com/pydantic/pydantic) — Data models
- [click](https://github.com/pallets/click) — CLI framework

## License

MIT
