# RPG Game Master — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MVP terminal RPG game master that ingests a PDF, extracts world entities via Claude, and lets the player explore the world through RAG-grounded narration.

**Architecture:** Two-phase system — an ingestion CLI that processes a book into structured world data + ChromaDB embeddings, and a game loop that uses RAG + Claude to narrate exploration. Rich library for terminal UI, SQLite for game state.

**Tech Stack:** Python 3.12, uv, anthropic, chromadb, sentence-transformers, pymupdf, rich, pydantic, click, SQLite

**Prerequisites:** Set `ANTHROPIC_API_KEY` environment variable before running ingestion or game.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/rpg_gm/__init__.py`
- Create: `src/rpg_gm/cli.py`
- Create: `src/rpg_gm/ingestion/__init__.py`
- Create: `src/rpg_gm/world/__init__.py`
- Create: `src/rpg_gm/game/__init__.py`
- Create: `src/rpg_gm/ui/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize uv project**

```bash
cd /home/fares/projects/rpg-game-master
uv init --lib --name rpg-gm
```

Then replace the generated `pyproject.toml` with:

```toml
[project]
name = "rpg-gm"
version = "0.1.0"
description = "Terminal RPG game master that turns books into explorable worlds"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.42.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "pymupdf>=1.24.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
]

[project.scripts]
rpg-gm = "rpg_gm.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create package structure**

Create all `__init__.py` files (empty) for:
- `src/rpg_gm/__init__.py`
- `src/rpg_gm/ingestion/__init__.py`
- `src/rpg_gm/world/__init__.py`
- `src/rpg_gm/game/__init__.py`
- `src/rpg_gm/ui/__init__.py`
- `tests/__init__.py`

Create a minimal `src/rpg_gm/cli.py`:

```python
import click


@click.group()
def main():
    """RPG Game Master — Turn any book into an explorable world."""
    pass


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", prompt="World name", help="Name for this world")
def ingest(file_path: str, name: str):
    """Ingest a PDF or text file into a playable world."""
    click.echo(f"Ingesting {file_path} as '{name}'...")


@main.command()
@click.argument("world_name")
def play(world_name: str):
    """Play an ingested world."""
    click.echo(f"Loading world '{world_name}'...")
```

**Step 3: Install dependencies and verify**

```bash
uv sync
uv run rpg-gm --help
```

Expected: Shows help text with `ingest` and `play` commands.

**Step 4: Add dev dependencies and verify pytest**

```bash
uv add --dev pytest
uv run pytest --co
```

Expected: "no tests ran" (no test files yet), no errors.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/
git commit -m "Scaffold project structure with CLI entry points"
```

---

### Task 2: World Data Models

**Files:**
- Create: `src/rpg_gm/world/models.py`
- Create: `tests/test_models.py`

**Step 1: Write tests for data models**

```python
# tests/test_models.py
import json
from rpg_gm.world.models import Location, NPC, Event, Faction, LoreEntry, World


def test_location_creation():
    loc = Location(
        id="agora",
        name="The Agora",
        description="The central marketplace of Athens.",
        connections=["acropolis", "piraeus-road"],
        details=["marble columns", "merchant stalls"],
        npcs=["socrates"],
        source_pages=[34, 35],
    )
    assert loc.id == "agora"
    assert "acropolis" in loc.connections
    assert len(loc.details) == 2


def test_npc_creation():
    npc = NPC(
        id="socrates",
        name="Socrates",
        role="philosopher",
        description="An Athenian philosopher known for his questioning method.",
        personality="Inquisitive, ironic, humble yet persistent in questioning.",
        knowledge=["philosophy", "ethics", "Athenian politics"],
        location_id="agora",
        relationships={"plato": "student and devoted follower"},
        source_pages=[12, 45, 67],
    )
    assert npc.id == "socrates"
    assert npc.role == "philosopher"
    assert "plato" in npc.relationships


def test_event_creation():
    event = Event(
        id="trial-of-socrates",
        name="The Trial of Socrates",
        description="Socrates was tried and sentenced to death in 399 BC.",
        time_period="399 BC",
        participants=["socrates"],
        locations=["agora"],
        source_pages=[100, 101],
    )
    assert event.time_period == "399 BC"


def test_faction_creation():
    faction = Faction(
        id="thirty-tyrants",
        name="The Thirty Tyrants",
        description="An oligarchic government imposed on Athens after the Peloponnesian War.",
        members=["critias"],
        goals="Maintain oligarchic rule over Athens",
        source_pages=[80],
    )
    assert faction.id == "thirty-tyrants"


def test_lore_entry_creation():
    lore = LoreEntry(
        id="athenian-democracy",
        category="politics",
        title="Athenian Democracy",
        content="Athens practiced direct democracy where citizens voted on legislation.",
        related_entities=["agora"],
        source_pages=[20, 21],
    )
    assert lore.category == "politics"


def test_world_serialization_roundtrip():
    loc = Location(
        id="agora",
        name="The Agora",
        description="Central marketplace.",
        connections=[],
        details=[],
        npcs=[],
        source_pages=[1],
    )
    npc = NPC(
        id="socrates",
        name="Socrates",
        role="philosopher",
        description="A philosopher.",
        personality="Inquisitive.",
        knowledge=["philosophy"],
        location_id="agora",
        relationships={},
        source_pages=[1],
    )
    world = World(
        title="Ancient Athens",
        source_file="athens.pdf",
        locations={"agora": loc},
        npcs={"socrates": npc},
        events={},
        factions={},
        lore={},
    )
    # Roundtrip through JSON
    data = json.loads(world.model_dump_json())
    world2 = World.model_validate(data)
    assert world2.title == "Ancient Athens"
    assert world2.locations["agora"].name == "The Agora"
    assert world2.npcs["socrates"].name == "Socrates"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rpg_gm.world.models'`

**Step 3: Implement data models**

```python
# src/rpg_gm/world/models.py
from pydantic import BaseModel


class Location(BaseModel):
    id: str
    name: str
    description: str
    connections: list[str] = []
    details: list[str] = []
    npcs: list[str] = []
    source_pages: list[int] = []


class NPC(BaseModel):
    id: str
    name: str
    role: str
    description: str
    personality: str
    knowledge: list[str] = []
    location_id: str
    relationships: dict[str, str] = {}
    source_pages: list[int] = []


class Event(BaseModel):
    id: str
    name: str
    description: str
    time_period: str
    participants: list[str] = []
    locations: list[str] = []
    source_pages: list[int] = []


class Faction(BaseModel):
    id: str
    name: str
    description: str
    members: list[str] = []
    goals: str
    source_pages: list[int] = []


class LoreEntry(BaseModel):
    id: str
    category: str
    title: str
    content: str
    related_entities: list[str] = []
    source_pages: list[int] = []


class World(BaseModel):
    title: str
    source_file: str
    locations: dict[str, Location] = {}
    npcs: dict[str, NPC] = {}
    events: dict[str, Event] = {}
    factions: dict[str, Faction] = {}
    lore: dict[str, LoreEntry] = {}
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/rpg_gm/world/models.py tests/test_models.py
git commit -m "Add Pydantic world data models"
```

---

### Task 3: World Loader (JSON persistence)

**Files:**
- Create: `src/rpg_gm/world/loader.py`
- Create: `tests/test_loader.py`

**Step 1: Write tests**

```python
# tests/test_loader.py
import json
from pathlib import Path

from rpg_gm.world.loader import save_world, load_world, get_world_dir, list_worlds
from rpg_gm.world.models import World, Location, NPC


def _make_test_world() -> World:
    loc = Location(
        id="agora",
        name="The Agora",
        description="Central marketplace.",
        connections=[],
        details=[],
        npcs=["socrates"],
        source_pages=[1],
    )
    npc = NPC(
        id="socrates",
        name="Socrates",
        role="philosopher",
        description="A philosopher.",
        personality="Inquisitive.",
        knowledge=["philosophy"],
        location_id="agora",
        relationships={},
        source_pages=[1],
    )
    return World(
        title="Ancient Athens",
        source_file="athens.pdf",
        locations={"agora": loc},
        npcs={"socrates": npc},
        events={},
        factions={},
        lore={},
    )


def test_get_world_dir(tmp_path):
    d = get_world_dir("ancient-athens", base_dir=tmp_path)
    assert d == tmp_path / "ancient-athens"


def test_save_and_load_world(tmp_path):
    world = _make_test_world()
    save_world(world, "ancient-athens", base_dir=tmp_path)

    world_file = tmp_path / "ancient-athens" / "world.json"
    assert world_file.exists()

    loaded = load_world("ancient-athens", base_dir=tmp_path)
    assert loaded.title == "Ancient Athens"
    assert loaded.locations["agora"].name == "The Agora"
    assert loaded.npcs["socrates"].name == "Socrates"


def test_list_worlds_empty(tmp_path):
    assert list_worlds(base_dir=tmp_path) == []


def test_list_worlds(tmp_path):
    world = _make_test_world()
    save_world(world, "ancient-athens", base_dir=tmp_path)
    save_world(world, "roman-empire", base_dir=tmp_path)

    worlds = list_worlds(base_dir=tmp_path)
    assert set(worlds) == {"ancient-athens", "roman-empire"}
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: FAIL — import error.

**Step 3: Implement world loader**

```python
# src/rpg_gm/world/loader.py
import json
from pathlib import Path

from rpg_gm.world.models import World

DEFAULT_WORLDS_DIR = Path("worlds")


def get_world_dir(world_name: str, base_dir: Path | None = None) -> Path:
    base = base_dir or DEFAULT_WORLDS_DIR
    return base / world_name


def save_world(world: World, world_name: str, base_dir: Path | None = None) -> Path:
    world_dir = get_world_dir(world_name, base_dir)
    world_dir.mkdir(parents=True, exist_ok=True)
    world_file = world_dir / "world.json"
    world_file.write_text(world.model_dump_json(indent=2))
    return world_file


def load_world(world_name: str, base_dir: Path | None = None) -> World:
    world_dir = get_world_dir(world_name, base_dir)
    world_file = world_dir / "world.json"
    if not world_file.exists():
        raise FileNotFoundError(f"World '{world_name}' not found at {world_file}")
    data = json.loads(world_file.read_text())
    return World.model_validate(data)


def list_worlds(base_dir: Path | None = None) -> list[str]:
    base = base_dir or DEFAULT_WORLDS_DIR
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "world.json").exists()
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/rpg_gm/world/loader.py tests/test_loader.py
git commit -m "Add world JSON save/load/list"
```

---

### Task 4: PDF Reader and Text Chunker

**Files:**
- Create: `src/rpg_gm/ingestion/reader.py`
- Create: `src/rpg_gm/ingestion/chunker.py`
- Create: `tests/test_reader.py`
- Create: `tests/test_chunker.py`

**Step 1: Write chunker tests**

```python
# tests/test_chunker.py
from rpg_gm.ingestion.chunker import chunk_text, Chunk


def test_chunk_short_text():
    """Text shorter than chunk size produces one chunk."""
    chunks = chunk_text("Hello world.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].index == 0


def test_chunk_splits_on_boundaries():
    """Long text is split into multiple chunks."""
    # Create text with clear sentence boundaries
    sentences = [f"Sentence number {i}." for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # All chunks should have text
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_chunk_overlap():
    """Adjacent chunks share overlapping text."""
    sentences = [f"This is sentence number {i} in the test." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # The end of chunk 0 should overlap with start of chunk 1
    # (overlap means some text appears in both)
    end_of_first = chunks[0].text[-50:]
    assert any(word in chunks[1].text for word in end_of_first.split() if len(word) > 3)


def test_chunk_page_numbers():
    """Chunks with page info carry page numbers."""
    pages = [(1, "Page one content here."), (2, "Page two content here.")]
    chunks = chunk_text(pages, chunk_size=100, overlap=10)
    assert all(isinstance(c.page, int) for c in chunks)


def test_chunk_dataclass():
    c = Chunk(index=0, text="hello", page=1)
    assert c.index == 0
    assert c.text == "hello"
    assert c.page == 1
```

**Step 2: Write reader tests**

```python
# tests/test_reader.py
from pathlib import Path
from rpg_gm.ingestion.reader import read_file


def test_read_text_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world.\nSecond line.")
    pages = read_file(str(f))
    # Text files return a single "page"
    assert len(pages) == 1
    assert pages[0][0] == 1  # page number
    assert "Hello world." in pages[0][1]


def test_read_nonexistent_file():
    try:
        read_file("/nonexistent/file.txt")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_chunker.py tests/test_reader.py -v
```

Expected: FAIL — import errors.

**Step 4: Implement Chunk and chunker**

```python
# src/rpg_gm/ingestion/chunker.py
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str
    page: int | None = None


def chunk_text(
    source: str | list[tuple[int, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Args:
        source: Either a plain string or a list of (page_number, text) tuples.
        chunk_size: Target chunk size in characters.
        overlap: Number of characters to overlap between chunks.
    """
    if isinstance(source, str):
        pages = [(None, source)]
    else:
        pages = source

    # Flatten into a single text with page markers
    segments: list[tuple[int | None, str]] = []
    for page_num, text in pages:
        for char in text:
            segments.append((page_num, char))

    full_text = "".join(c for _, c in segments)
    if not full_text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))

        # Try to break at a sentence boundary
        if end < len(full_text):
            search_start = max(start + chunk_size // 2, start)
            best_break = end
            for i in range(end, search_start, -1):
                if full_text[i - 1] in ".!?\n":
                    best_break = i
                    break
            end = best_break

        chunk_text_str = full_text[start:end].strip()
        if chunk_text_str:
            # Get page number from the midpoint of this chunk
            mid = (start + end) // 2
            page = segments[mid][0] if mid < len(segments) else None
            chunks.append(Chunk(index=idx, text=chunk_text_str, page=page))
            idx += 1

        start = end - overlap
        if start >= len(full_text) - overlap:
            break

    return chunks
```

**Step 5: Implement reader**

```python
# src/rpg_gm/ingestion/reader.py
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
```

**Step 6: Run tests**

```bash
uv run pytest tests/test_chunker.py tests/test_reader.py -v
```

Expected: All tests PASS.

**Step 7: Commit**

```bash
git add src/rpg_gm/ingestion/reader.py src/rpg_gm/ingestion/chunker.py tests/test_reader.py tests/test_chunker.py
git commit -m "Add PDF reader and text chunker"
```

---

### Task 5: ChromaDB Embedder

**Files:**
- Create: `src/rpg_gm/ingestion/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write tests**

```python
# tests/test_embedder.py
from pathlib import Path

from rpg_gm.ingestion.chunker import Chunk
from rpg_gm.ingestion.embedder import embed_chunks, query_chunks


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(index=0, text="Athens was a powerful city-state in ancient Greece.", page=1),
        Chunk(index=1, text="The Parthenon sits atop the Acropolis hill.", page=5),
        Chunk(index=2, text="Socrates spent his days questioning people in the Agora.", page=12),
        Chunk(index=3, text="The Roman Empire expanded across the Mediterranean.", page=20),
    ]


def test_embed_chunks(tmp_path):
    chunks = _sample_chunks()
    collection = embed_chunks(chunks, "test-world", persist_dir=tmp_path)
    assert collection.count() == 4


def test_query_returns_relevant_results(tmp_path):
    chunks = _sample_chunks()
    embed_chunks(chunks, "test-world", persist_dir=tmp_path)
    results = query_chunks("philosophy in Athens", "test-world", persist_dir=tmp_path, n_results=2)
    assert len(results) == 2
    # The Socrates chunk should be among top results
    texts = [r["text"] for r in results]
    assert any("Socrates" in t or "Athens" in t for t in texts)


def test_query_includes_metadata(tmp_path):
    chunks = _sample_chunks()
    embed_chunks(chunks, "test-world", persist_dir=tmp_path)
    results = query_chunks("Parthenon", "test-world", persist_dir=tmp_path, n_results=1)
    assert len(results) == 1
    assert "page" in results[0]
    assert "chunk_index" in results[0]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_embedder.py -v
```

Expected: FAIL — import error.

**Step 3: Implement embedder**

```python
# src/rpg_gm/ingestion/embedder.py
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rpg_gm.ingestion.chunker import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    if persist_dir:
        return chromadb.PersistentClient(path=str(persist_dir))
    return chromadb.PersistentClient(path="worlds/_chroma")


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=DEFAULT_MODEL)


def embed_chunks(
    chunks: list[Chunk],
    world_name: str,
    persist_dir: Path | None = None,
) -> chromadb.Collection:
    """Embed chunks into a ChromaDB collection."""
    client = _get_client(persist_dir)
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=world_name.replace(" ", "-"),
        embedding_function=ef,
    )

    # Batch insert
    ids = [f"chunk-{c.index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "chunk_index": c.index,
            "page": c.page or 0,
            "preview": c.text[:100],
        }
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def query_chunks(
    query: str,
    world_name: str,
    persist_dir: Path | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Query the ChromaDB collection and return relevant chunks."""
    client = _get_client(persist_dir)
    ef = _get_embedding_fn()
    collection = client.get_collection(
        name=world_name.replace(" ", "-"),
        embedding_function=ef,
    )

    results = collection.query(query_texts=[query], n_results=n_results)

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i].get("page"),
            "chunk_index": results["metadatas"][0][i].get("chunk_index"),
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return output
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_embedder.py -v
```

Expected: All 3 tests PASS. (First run will download all-MiniLM-L6-v2 model ~80MB.)

**Step 5: Commit**

```bash
git add src/rpg_gm/ingestion/embedder.py tests/test_embedder.py
git commit -m "Add ChromaDB embedding and retrieval"
```

---

### Task 6: Claude Entity Extractor

**Files:**
- Create: `src/rpg_gm/ingestion/extractor.py`
- Create: `tests/test_extractor.py`

**Step 1: Write tests**

The extractor calls Claude, so tests use a mock. Test the prompt assembly and response parsing, not the API call.

```python
# tests/test_extractor.py
from unittest.mock import MagicMock, patch
import json

from rpg_gm.ingestion.extractor import (
    extract_entities_from_chunks,
    parse_extraction_response,
    build_extraction_prompt,
)
from rpg_gm.ingestion.chunker import Chunk


def test_build_extraction_prompt():
    chunks = [
        Chunk(index=0, text="The Agora was the central marketplace of Athens.", page=1),
        Chunk(index=1, text="Socrates often debated in the Agora.", page=2),
    ]
    prompt = build_extraction_prompt(chunks, existing_names=[])
    assert "Agora" in prompt
    assert "Socrates" in prompt
    assert "locations" in prompt.lower()


def test_build_extraction_prompt_includes_existing():
    chunks = [Chunk(index=0, text="Some text.", page=1)]
    prompt = build_extraction_prompt(chunks, existing_names=["agora", "socrates"])
    assert "agora" in prompt.lower()
    assert "socrates" in prompt.lower()


def test_parse_extraction_response_valid():
    raw = {
        "locations": [
            {
                "name": "The Agora",
                "description": "The central marketplace.",
                "connections": [],
                "details": ["merchant stalls"],
                "source_pages": [1],
            }
        ],
        "npcs": [
            {
                "name": "Socrates",
                "role": "philosopher",
                "description": "A famous philosopher.",
                "personality": "Inquisitive and ironic.",
                "knowledge": ["philosophy", "ethics"],
                "location": "The Agora",
                "relationships": {},
                "source_pages": [2],
            }
        ],
        "events": [],
        "factions": [],
        "lore": [],
    }
    entities = parse_extraction_response(raw)
    assert len(entities["locations"]) == 1
    assert len(entities["npcs"]) == 1
    assert entities["locations"][0]["name"] == "The Agora"


def test_parse_extraction_response_empty():
    raw = {"locations": [], "npcs": [], "events": [], "factions": [], "lore": []}
    entities = parse_extraction_response(raw)
    assert all(len(v) == 0 for v in entities.values())
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extractor.py -v
```

Expected: FAIL — import error.

**Step 3: Implement extractor**

```python
# src/rpg_gm/ingestion/extractor.py
import json
import re

import anthropic

from rpg_gm.ingestion.chunker import Chunk

EXTRACTION_SCHEMA = {
    "locations": [
        {
            "name": "string",
            "description": "string",
            "connections": ["other location names"],
            "details": ["examinable details"],
            "source_pages": [0],
        }
    ],
    "npcs": [
        {
            "name": "string",
            "role": "string (e.g. philosopher, merchant, ruler)",
            "description": "string",
            "personality": "brief personality description for roleplay",
            "knowledge": ["topics this person knows about"],
            "location": "location name where they are found",
            "relationships": {"other_npc_name": "relationship description"},
            "source_pages": [0],
        }
    ],
    "events": [
        {
            "name": "string",
            "description": "string",
            "time_period": "string",
            "participants": ["names"],
            "locations": ["location names"],
            "source_pages": [0],
        }
    ],
    "factions": [
        {
            "name": "string",
            "description": "string",
            "members": ["names"],
            "goals": "string",
            "source_pages": [0],
        }
    ],
    "lore": [
        {
            "category": "culture|religion|economy|geography|politics|science|other",
            "title": "string",
            "content": "string",
            "related_entities": ["names of related locations/npcs/events"],
            "source_pages": [0],
        }
    ],
}


def build_extraction_prompt(
    chunks: list[Chunk], existing_names: list[str]
) -> str:
    chunk_texts = "\n\n".join(
        f"[Page {c.page or '?'}, Chunk {c.index}]\n{c.text}" for c in chunks
    )
    existing_note = ""
    if existing_names:
        existing_note = (
            f"\n\nAlready extracted entities (avoid duplicates): "
            f"{', '.join(existing_names)}"
        )

    return f"""Analyze the following passages from a book and extract structured world data.

Extract all locations, characters/NPCs, historical events, factions/groups, and cultural/lore details you can find.

For each entity, include the source page numbers where the information was found.

Only extract entities that are clearly described in the text. Do not invent or speculate.{existing_note}

Return your response as JSON matching this exact schema:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}

If a category has no entities in these passages, use an empty list.

--- SOURCE PASSAGES ---
{chunk_texts}
--- END PASSAGES ---

Respond with ONLY the JSON object, no other text."""


def parse_extraction_response(raw: dict) -> dict:
    """Parse and validate extraction response."""
    result = {
        "locations": [],
        "npcs": [],
        "events": [],
        "factions": [],
        "lore": [],
    }
    for key in result:
        if key in raw and isinstance(raw[key], list):
            result[key] = raw[key]
    return result


def extract_entities_from_chunks(
    chunks: list[Chunk],
    existing_names: list[str] | None = None,
) -> dict:
    """Call Claude to extract world entities from text chunks."""
    client = anthropic.Anthropic()
    prompt = build_extraction_prompt(chunks, existing_names or [])

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)

    raw = json.loads(response_text)
    return parse_extraction_response(raw)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_extractor.py -v
```

Expected: All 4 tests PASS. (These tests don't call the API — they test prompt building and response parsing.)

**Step 5: Commit**

```bash
git add src/rpg_gm/ingestion/extractor.py tests/test_extractor.py
git commit -m "Add Claude entity extraction with prompt and parser"
```

---

### Task 7: Rich Terminal UI Display

**Files:**
- Create: `src/rpg_gm/ui/display.py`

No tests for this task — it's pure presentation. We'll verify visually.

**Step 1: Implement display module**

```python
# src/rpg_gm/ui/display.py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

from rpg_gm.world.models import Location, NPC, World

console = Console()


def show_title(world_title: str):
    console.print()
    console.print(
        Panel(
            f"[bold]{world_title}[/bold]",
            style="bold cyan",
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )
    console.print()


def show_location(location: Location, world: World):
    # Location name and description
    content = f"[italic]{location.description}[/italic]"

    # Exits
    if location.connections:
        exits = []
        for conn_id in location.connections:
            conn = world.locations.get(conn_id)
            exits.append(conn.name if conn else conn_id)
        content += f"\n\n[dim]Exits:[/dim] {', '.join(exits)}"

    # NPCs present
    if location.npcs:
        npc_names = []
        for npc_id in location.npcs:
            npc = world.npcs.get(npc_id)
            npc_names.append(npc.name if npc else npc_id)
        content += f"\n[dim]People here:[/dim] {', '.join(npc_names)}"

    # Examinable details
    if location.details:
        content += f"\n[dim]You notice:[/dim] {', '.join(location.details)}"

    console.print(Panel(content, title=f"[bold green]{location.name}[/bold green]", box=box.ROUNDED))


def show_narration(text: str):
    console.print()
    console.print(Panel(text, style="white", box=box.SIMPLE))


def show_npc_dialogue(npc_name: str, text: str):
    console.print()
    console.print(
        Panel(text, title=f"[bold yellow]{npc_name}[/bold yellow]", border_style="yellow", box=box.ROUNDED)
    )


def show_journal(entries: list[dict]):
    if not entries:
        console.print("[dim]Your journal is empty.[/dim]")
        return

    table = Table(title="Journal", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim", width=20)
    table.add_column("Entry")
    for entry in entries:
        table.add_row(entry.get("time", ""), entry.get("text", ""))
    console.print(table)


def show_map(visited: set[str], world: World, current_id: str):
    table = Table(title="Known Locations", box=box.ROUNDED)
    table.add_column("Location", style="bold")
    table.add_column("Connections")
    table.add_column("Status")

    for loc_id in sorted(visited):
        loc = world.locations.get(loc_id)
        if not loc:
            continue
        connections = []
        for conn_id in loc.connections:
            conn = world.locations.get(conn_id)
            name = conn.name if conn else conn_id
            if conn_id in visited:
                connections.append(f"[green]{name}[/green]")
            else:
                connections.append(f"[dim]{name}[/dim]")
        marker = "[bold cyan]<< You are here[/bold cyan]" if loc_id == current_id else "[green]Visited[/green]"
        table.add_row(loc.name, ", ".join(connections) if connections else "[dim]none[/dim]", marker)

    console.print(table)


def show_discoveries(discovered: dict, world: World):
    table = Table(title="Discoveries", box=box.ROUNDED)
    table.add_column("Category", style="bold")
    table.add_column("Found")
    table.add_column("Total")
    table.add_column("Progress")

    categories = [
        ("Locations", len(discovered.get("locations", set())), len(world.locations)),
        ("Characters", len(discovered.get("npcs", set())), len(world.npcs)),
        ("Events", len(discovered.get("events", set())), len(world.events)),
        ("Factions", len(discovered.get("factions", set())), len(world.factions)),
        ("Lore", len(discovered.get("lore", set())), len(world.lore)),
    ]

    for name, found, total in categories:
        pct = (found / total * 100) if total > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_len - filled)
        table.add_row(name, str(found), str(total), f"{bar} {pct:.0f}%")

    console.print(table)


def show_entity_for_review(entity_type: str, entity: dict):
    """Show an extracted entity for user review during ingestion."""
    content = ""
    for key, value in entity.items():
        if isinstance(value, list) and value:
            content += f"[bold]{key}:[/bold] {', '.join(str(v) for v in value)}\n"
        elif isinstance(value, dict) and value:
            items = [f"{k}: {v}" for k, v in value.items()]
            content += f"[bold]{key}:[/bold] {'; '.join(items)}\n"
        elif value:
            content += f"[bold]{key}:[/bold] {value}\n"

    title = f"New {entity_type.rstrip('s').title()}"
    console.print(Panel(content.strip(), title=f"[bold magenta]{title}[/bold magenta]", box=box.ROUNDED))
    console.print("[dim][a]ccept  [s]kip  [q]uit extraction[/dim]")


def show_error(msg: str):
    console.print(f"[bold red]Error:[/bold red] {msg}")


def show_info(msg: str):
    console.print(f"[dim]{msg}[/dim]")


def show_help():
    table = Table(title="Commands", box=box.SIMPLE)
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/look", "Describe your current location in detail")
    table.add_row("/map", "Show visited locations and connections")
    table.add_row("/talk <name>", "Start a conversation with someone")
    table.add_row("/examine <thing>", "Examine something in detail")
    table.add_row("/journal", "View your journal")
    table.add_row("/discoveries", "See what you've uncovered")
    table.add_row("/save", "Save your game")
    table.add_row("/load", "Load a saved game")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit the game")
    table.add_row("[dim]anything else[/dim]", "[dim]Describe what you want to do[/dim]")
    console.print(table)


def get_input(prompt_text: str = "> ") -> str:
    try:
        return console.input(f"[bold cyan]{prompt_text}[/bold cyan]").strip()
    except (EOFError, KeyboardInterrupt):
        return "/quit"
```

**Step 2: Commit**

```bash
git add src/rpg_gm/ui/display.py
git commit -m "Add Rich terminal UI display module"
```

---

### Task 8: SQLite Game State

**Files:**
- Create: `src/rpg_gm/game/state.py`
- Create: `tests/test_state.py`

**Step 1: Write tests**

```python
# tests/test_state.py
from rpg_gm.game.state import GameState


def test_create_new_state(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    assert state.get_location() == "agora"
    assert state.get_visited() == {"agora"}
    state.close()


def test_move_location(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    state.set_location("acropolis")
    assert state.get_location() == "acropolis"
    assert state.get_visited() == {"agora", "acropolis"}
    state.close()


def test_journal(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    state.add_journal_entry("Arrived at the Agora.")
    state.add_journal_entry("Met Socrates.")
    entries = state.get_journal()
    assert len(entries) == 2
    assert entries[0]["text"] == "Arrived at the Agora."
    state.close()


def test_discovered_entities(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    state.discover("locations", "agora")
    state.discover("npcs", "socrates")
    disc = state.get_discovered()
    assert "agora" in disc["locations"]
    assert "socrates" in disc["npcs"]
    state.close()


def test_conversation_history(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    state.add_message("user", "Hello Socrates")
    state.add_message("assistant", "Greetings, friend.")
    history = state.get_recent_messages(limit=10)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    state.close()


def test_conversation_history_limit(tmp_path):
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("ancient-athens", "agora")
    for i in range(25):
        state.add_message("user", f"Message {i}")
    history = state.get_recent_messages(limit=20)
    assert len(history) == 20
    # Should be the most recent 20
    assert history[-1]["content"] == "Message 24"
    state.close()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_state.py -v
```

Expected: FAIL — import error.

**Step 3: Implement game state**

```python
# src/rpg_gm/game/state.py
import json
import sqlite3
from datetime import datetime


class GameState:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discovered (
                category TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (category, entity_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def init(self, world_name: str, start_location: str):
        """Initialize a new game or load existing state."""
        self._set("world_name", world_name)
        if not self._get("location"):
            self._set("location", start_location)
            self._add_visited(start_location)

    def _set(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def _get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_location(self) -> str:
        return self._get("location") or ""

    def set_location(self, location_id: str):
        self._set("location", location_id)
        self._add_visited(location_id)

    def _add_visited(self, location_id: str):
        visited = self.get_visited()
        visited.add(location_id)
        self._set("visited", json.dumps(sorted(visited)))

    def get_visited(self) -> set[str]:
        raw = self._get("visited")
        if not raw:
            return set()
        return set(json.loads(raw))

    def add_journal_entry(self, text: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.conn.execute(
            "INSERT INTO journal (timestamp, text) VALUES (?, ?)",
            (now, text),
        )
        self.conn.commit()

    def get_journal(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, text FROM journal ORDER BY id"
        ).fetchall()
        return [{"time": r["timestamp"], "text": r["text"]} for r in rows]

    def discover(self, category: str, entity_id: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO discovered (category, entity_id, timestamp) VALUES (?, ?, ?)",
            (category, entity_id, now),
        )
        self.conn.commit()

    def get_discovered(self) -> dict[str, set[str]]:
        rows = self.conn.execute("SELECT category, entity_id FROM discovered").fetchall()
        result: dict[str, set[str]] = {
            "locations": set(),
            "npcs": set(),
            "events": set(),
            "factions": set(),
            "lore": set(),
        }
        for r in rows:
            cat = r["category"]
            if cat not in result:
                result[cat] = set()
            result[cat].add(r["entity_id"])
        return result

    def add_message(self, role: str, content: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, now),
        )
        self.conn.commit()

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def close(self):
        self.conn.close()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_state.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/rpg_gm/game/state.py tests/test_state.py
git commit -m "Add SQLite game state management"
```

---

### Task 9: Claude Narrator (RAG-grounded)

**Files:**
- Create: `src/rpg_gm/game/narrator.py`
- Create: `tests/test_narrator.py`

**Step 1: Write tests for prompt assembly**

```python
# tests/test_narrator.py
from rpg_gm.game.narrator import build_gm_prompt, build_npc_prompt
from rpg_gm.world.models import Location, NPC, World


def _make_world() -> World:
    return World(
        title="Ancient Athens",
        source_file="athens.pdf",
        locations={
            "agora": Location(
                id="agora",
                name="The Agora",
                description="The central marketplace of Athens.",
                connections=["acropolis"],
                details=["marble columns"],
                npcs=["socrates"],
                source_pages=[1],
            )
        },
        npcs={
            "socrates": NPC(
                id="socrates",
                name="Socrates",
                role="philosopher",
                description="A famous Athenian philosopher.",
                personality="Inquisitive, ironic, humble.",
                knowledge=["philosophy", "ethics"],
                location_id="agora",
                relationships={},
                source_pages=[1],
            )
        },
        events={},
        factions={},
        lore={},
    )


def test_build_gm_prompt():
    world = _make_world()
    passages = [{"text": "The Agora was bustling with merchants.", "page": 5}]
    history = [{"role": "user", "content": "I look around."}]
    prompt = build_gm_prompt(
        world=world,
        location_id="agora",
        passages=passages,
        history=history,
        player_input="I walk toward the columns.",
    )
    assert "Ancient Athens" in prompt["system"]
    assert "The Agora" in prompt["system"]
    assert "Socrates" in prompt["system"]
    assert "bustling with merchants" in prompt["messages"][-1]["content"]
    assert "walk toward the columns" in prompt["messages"][-1]["content"]


def test_build_npc_prompt():
    world = _make_world()
    npc = world.npcs["socrates"]
    passages = [{"text": "Socrates believed virtue was knowledge.", "page": 10}]
    history = []
    prompt = build_npc_prompt(
        npc=npc,
        passages=passages,
        history=history,
        player_input="What is virtue?",
    )
    assert "Socrates" in prompt["system"]
    assert "Inquisitive" in prompt["system"]
    assert "philosophy" in prompt["system"]
    assert "virtue was knowledge" in prompt["messages"][-1]["content"]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_narrator.py -v
```

Expected: FAIL — import error.

**Step 3: Implement narrator**

```python
# src/rpg_gm/game/narrator.py
import json
import re

import anthropic

from rpg_gm.world.models import World, NPC


GM_SYSTEM_TEMPLATE = """You are an immersive game master guiding a player through {world_title}.
You narrate scenes, describe locations, and bring the world to life.

Rules:
- Stay faithful to the source material. Never invent facts not supported by the provided passages.
- When describing a location or event, weave in specific details from the source passages provided in context.
- Write in second person present tense ("You see...", "You hear...").
- Keep responses to 2-4 paragraphs. Rich in sensory detail but not verbose.
- If the player tries to go somewhere that doesn't exist in the world, gently redirect them toward known locations.
- End each narration with subtle hooks — things to examine, people to talk to, paths to explore.

IMPORTANT — State change tags:
When the player's action results in moving to a new location, include this JSON on its own line at the very end of your response:
{{"move_to": "location_id"}}

When the player discovers something significant (a new fact, event, or meets someone), include:
{{"discover": {{"type": "lore|event|npc", "id": "entity_id"}}}}

These tags are parsed by the game engine. Always include them when applicable.

--- CURRENT WORLD STATE ---
Location: {location_name} — {location_description}
{exits}
{npcs_present}
"""

NPC_SYSTEM_TEMPLATE = """You are {npc_name}, {npc_role}. {npc_description}

Personality: {npc_personality}

You know about: {npc_knowledge}

Rules:
- Stay in character. Speak as this person would — with their vocabulary, concerns, and worldview.
- You can discuss these topics: {npc_knowledge}. For topics outside your knowledge, say you don't know or redirect the conversation.
- Reference specific details from the source passages when answering questions.
- Do not break character. Do not acknowledge being an AI or a game.
- Keep responses to 1-3 paragraphs of natural dialogue.
- When the player wants to end the conversation, include on its own line at the end:
{{"end_conversation": true}}
"""


def build_gm_prompt(
    world: World,
    location_id: str,
    passages: list[dict],
    history: list[dict],
    player_input: str,
) -> dict:
    location = world.locations[location_id]

    exits = "Exits: " + ", ".join(
        world.locations[c].name if c in world.locations else c
        for c in location.connections
    ) if location.connections else "No obvious exits."

    npc_names = []
    for npc_id in location.npcs:
        npc = world.npcs.get(npc_id)
        if npc:
            npc_names.append(f"{npc.name} ({npc.role})")
    npcs_present = "People here: " + ", ".join(npc_names) if npc_names else "Nobody else is here."

    system = GM_SYSTEM_TEMPLATE.format(
        world_title=world.title,
        location_name=location.name,
        location_description=location.description,
        exits=exits,
        npcs_present=npcs_present,
    )

    # Build the user message with RAG context
    passage_text = ""
    if passages:
        passage_text = "\n\nRELEVANT SOURCE PASSAGES:\n"
        for p in passages:
            page_ref = f" (p.{p['page']})" if p.get("page") else ""
            passage_text += f"- {p['text']}{page_ref}\n"

    user_content = f"{passage_text}\n\nPLAYER ACTION: {player_input}"

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    return {"system": system, "messages": messages}


def build_npc_prompt(
    npc: NPC,
    passages: list[dict],
    history: list[dict],
    player_input: str,
) -> dict:
    system = NPC_SYSTEM_TEMPLATE.format(
        npc_name=npc.name,
        npc_role=npc.role,
        npc_description=npc.description,
        npc_personality=npc.personality,
        npc_knowledge=", ".join(npc.knowledge),
    )

    passage_text = ""
    if passages:
        passage_text = "\n\nRELEVANT SOURCE PASSAGES:\n"
        for p in passages:
            page_ref = f" (p.{p['page']})" if p.get("page") else ""
            passage_text += f"- {p['text']}{page_ref}\n"

    user_content = f"{passage_text}\n\nThe player says: {player_input}"

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    return {"system": system, "messages": messages}


def narrate(prompt: dict) -> str:
    """Send prompt to Claude and return the full response text."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=prompt["system"],
        messages=prompt["messages"],
    )
    return message.content[0].text


def narrate_stream(prompt: dict):
    """Send prompt to Claude and yield text chunks as they arrive."""
    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=prompt["system"],
        messages=prompt["messages"],
    ) as stream:
        for text in stream.text_stream:
            yield text


def parse_state_changes(response_text: str) -> list[dict]:
    """Extract JSON state change tags from narrator response."""
    changes = []
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "move_to" in data or "discover" in data or "end_conversation" in data:
                    changes.append(data)
            except json.JSONDecodeError:
                continue
    return changes


def strip_state_tags(response_text: str) -> str:
    """Remove JSON state change tags from response for display."""
    lines = []
    for line in response_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                if "move_to" in data or "discover" in data or "end_conversation" in data:
                    continue
            except json.JSONDecodeError:
                pass
        lines.append(line)
    return "\n".join(lines).strip()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_narrator.py -v
```

Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/rpg_gm/game/narrator.py tests/test_narrator.py
git commit -m "Add Claude narrator with RAG prompt assembly"
```

---

### Task 10: Game Commands

**Files:**
- Create: `src/rpg_gm/game/commands.py`

**Step 1: Implement command parser and handlers**

```python
# src/rpg_gm/game/commands.py
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    name: str
    args: str = ""


def parse_input(raw: str) -> ParsedCommand | None:
    """Parse player input. Returns a ParsedCommand if it's a slash command, None if free text."""
    raw = raw.strip()
    if not raw.startswith("/"):
        return None
    parts = raw[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return ParsedCommand(name=name, args=args)


def find_npc_by_name(name: str, npc_ids: list[str], world) -> str | None:
    """Fuzzy-match an NPC name to an ID. Returns the NPC ID or None."""
    name_lower = name.lower()
    for npc_id in npc_ids:
        npc = world.npcs.get(npc_id)
        if npc and name_lower in npc.name.lower():
            return npc_id
    return None


def find_location_by_name(name: str, location_ids: list[str], world) -> str | None:
    """Fuzzy-match a location name to an ID. Returns the location ID or None."""
    name_lower = name.lower()
    for loc_id in location_ids:
        loc = world.locations.get(loc_id)
        if loc and name_lower in loc.name.lower():
            return loc_id
    # Also try matching the ID directly
    for loc_id in location_ids:
        if name_lower in loc_id.lower():
            return loc_id
    return None
```

**Step 2: Commit**

```bash
git add src/rpg_gm/game/commands.py
git commit -m "Add command parser and entity name matching"
```

---

### Task 11: Game Engine (Main Loop)

**Files:**
- Create: `src/rpg_gm/game/engine.py`

This is the core gameplay loop. It ties together all the components.

**Step 1: Implement the game engine**

```python
# src/rpg_gm/game/engine.py
from pathlib import Path

from rpg_gm.world.models import World
from rpg_gm.world.loader import load_world, get_world_dir
from rpg_gm.game.state import GameState
from rpg_gm.game.commands import parse_input, find_npc_by_name, find_location_by_name
from rpg_gm.game.narrator import (
    build_gm_prompt,
    build_npc_prompt,
    narrate,
    narrate_stream,
    parse_state_changes,
    strip_state_tags,
)
from rpg_gm.ingestion.embedder import query_chunks
from rpg_gm.ui.display import (
    show_title,
    show_location,
    show_narration,
    show_npc_dialogue,
    show_journal,
    show_map,
    show_discoveries,
    show_help,
    show_error,
    show_info,
    get_input,
    console,
)


class GameEngine:
    def __init__(self, world_name: str, base_dir: Path | None = None):
        self.world_name = world_name
        self.base_dir = base_dir or Path("worlds")
        self.world = load_world(world_name, base_dir=self.base_dir)
        self.world_dir = get_world_dir(world_name, base_dir=self.base_dir)

        # Set up game state
        saves_dir = self.world_dir / "saves"
        saves_dir.mkdir(exist_ok=True)
        self.state = GameState(str(saves_dir / "autosave.db"))

        # Pick a starting location (first location in the world)
        start = next(iter(self.world.locations))
        self.state.init(world_name, start)

        # Track conversation mode
        self.talking_to: str | None = None  # NPC ID if in conversation

    def run(self):
        """Main game loop."""
        show_title(self.world.title)
        self._discover_current_location()
        self._show_current_location()
        show_info("Type /help for commands, or describe what you want to do.")
        console.print()

        while True:
            prompt_text = f"[{self.world.npcs[self.talking_to].name}] > " if self.talking_to else "> "
            raw_input = get_input(prompt_text)

            if not raw_input:
                continue

            cmd = parse_input(raw_input)

            if cmd:
                should_quit = self._handle_command(cmd)
                if should_quit:
                    break
            else:
                self._handle_free_input(raw_input)

        show_info("Game saved. Farewell, traveler.")
        self.state.close()

    def _handle_command(self, cmd) -> bool:
        """Handle a slash command. Returns True if the game should quit."""
        if cmd.name == "quit":
            return True
        elif cmd.name == "help":
            show_help()
        elif cmd.name == "look":
            self._do_look()
        elif cmd.name == "map":
            show_map(self.state.get_visited(), self.world, self.state.get_location())
        elif cmd.name == "talk":
            self._do_talk(cmd.args)
        elif cmd.name == "journal":
            show_journal(self.state.get_journal())
        elif cmd.name == "discoveries":
            show_discoveries(self.state.get_discovered(), self.world)
        elif cmd.name == "examine":
            self._do_examine(cmd.args)
        elif cmd.name == "save":
            show_info("Game auto-saves continuously. Your progress is safe.")
        elif cmd.name == "load":
            show_info("Game loaded from last autosave on startup.")
        elif cmd.name in ("end", "bye", "leave"):
            if self.talking_to:
                npc = self.world.npcs[self.talking_to]
                show_info(f"You end your conversation with {npc.name}.")
                self.talking_to = None
            else:
                show_info("You're not in a conversation.")
        else:
            show_error(f"Unknown command: /{cmd.name}. Type /help for available commands.")
        return False

    def _handle_free_input(self, player_input: str):
        """Handle free-text input — either conversation or exploration."""
        if self.talking_to:
            self._do_npc_conversation(player_input)
        else:
            self._do_explore(player_input)

    def _do_look(self):
        """Detailed look at current location using RAG."""
        location_id = self.state.get_location()
        location = self.world.locations[location_id]
        passages = self._query_rag(f"description of {location.name}")
        prompt = build_gm_prompt(
            world=self.world,
            location_id=location_id,
            passages=passages,
            history=[],
            player_input=f"Describe {location.name} in vivid detail. What do I see, hear, and smell?",
        )
        self._stream_narration(prompt)

    def _do_talk(self, npc_name: str):
        """Start a conversation with an NPC."""
        if not npc_name:
            location = self.world.locations[self.state.get_location()]
            if location.npcs:
                npc_names = [self.world.npcs[nid].name for nid in location.npcs if nid in self.world.npcs]
                show_info(f"People here: {', '.join(npc_names)}")
            else:
                show_info("There's nobody here to talk to.")
            return

        location = self.world.locations[self.state.get_location()]
        npc_id = find_npc_by_name(npc_name, location.npcs, self.world)

        if not npc_id:
            # Check if NPC exists elsewhere
            all_npc_id = find_npc_by_name(npc_name, list(self.world.npcs.keys()), self.world)
            if all_npc_id:
                npc = self.world.npcs[all_npc_id]
                show_error(f"{npc.name} is not here. They can be found at {npc.location_id}.")
            else:
                show_error(f"Nobody named '{npc_name}' is known to you.")
            return

        self.talking_to = npc_id
        npc = self.world.npcs[npc_id]
        self.state.discover("npcs", npc_id)
        self.state.add_journal_entry(f"Began speaking with {npc.name}.")
        show_info(f"You approach {npc.name}.")

        # Opening greeting from NPC
        passages = self._query_rag(f"{npc.name} {npc.role}")
        prompt = build_npc_prompt(
            npc=npc,
            passages=passages,
            history=[],
            player_input="I approach you and greet you.",
        )
        self._stream_npc_dialogue(npc.name, prompt)

    def _do_examine(self, thing: str):
        """Examine something in the current location."""
        if not thing:
            show_info("Examine what? Try /examine <something>")
            return
        location = self.world.locations[self.state.get_location()]
        passages = self._query_rag(f"{thing} at {location.name}")
        prompt = build_gm_prompt(
            world=self.world,
            location_id=self.state.get_location(),
            passages=passages,
            history=[],
            player_input=f"I examine {thing} closely. Describe it in detail.",
        )
        self._stream_narration(prompt)

    def _do_explore(self, player_input: str):
        """Handle free-text exploration input."""
        location_id = self.state.get_location()
        passages = self._query_rag(player_input)
        history = self.state.get_recent_messages(limit=10)
        prompt = build_gm_prompt(
            world=self.world,
            location_id=location_id,
            passages=passages,
            history=history,
            player_input=player_input,
        )
        response = self._collect_narration(prompt)
        self.state.add_message("user", player_input)
        self.state.add_message("assistant", response)
        self._process_state_changes(response)

    def _do_npc_conversation(self, player_input: str):
        """Handle free-text input during NPC conversation."""
        npc = self.world.npcs[self.talking_to]
        passages = self._query_rag(f"{npc.name} {player_input}")
        history = self.state.get_recent_messages(limit=10)
        prompt = build_npc_prompt(
            npc=npc,
            passages=passages,
            history=history,
            player_input=player_input,
        )
        response = self._collect_npc_dialogue(npc.name, prompt)
        self.state.add_message("user", player_input)
        self.state.add_message("assistant", response)

        # Check for end conversation
        changes = parse_state_changes(response)
        for change in changes:
            if change.get("end_conversation"):
                show_info(f"{npc.name} turns away. The conversation has ended.")
                self.talking_to = None
                return
            if change.get("discover"):
                disc = change["discover"]
                self.state.discover(disc["type"], disc["id"])

    def _query_rag(self, query: str) -> list[dict]:
        """Query ChromaDB for relevant passages."""
        try:
            return query_chunks(
                query=query,
                world_name=self.world_name,
                persist_dir=self.world_dir / "chroma",
                n_results=5,
            )
        except Exception:
            return []

    def _stream_narration(self, prompt: dict):
        """Stream narration and display it."""
        full_text = ""
        console.print()
        try:
            for chunk in narrate_stream(prompt):
                console.print(chunk, end="")
                full_text += chunk
        except Exception as e:
            show_error(f"Narration failed: {e}")
            return
        console.print()
        console.print()
        clean = strip_state_tags(full_text)
        self._process_state_changes(full_text)

    def _collect_narration(self, prompt: dict) -> str:
        """Stream narration, display it, and return full text."""
        full_text = ""
        console.print()
        try:
            for chunk in narrate_stream(prompt):
                console.print(chunk, end="")
                full_text += chunk
        except Exception as e:
            show_error(f"Narration failed: {e}")
            return ""
        console.print()
        console.print()
        # Display clean version (without tags) would require buffering.
        # For MVP, we print as-is (tags are on their own lines, not too distracting).
        return full_text

    def _stream_npc_dialogue(self, npc_name: str, prompt: dict):
        """Stream NPC dialogue and display it."""
        full_text = ""
        console.print()
        console.print(f"[bold yellow]{npc_name}:[/bold yellow]")
        try:
            for chunk in narrate_stream(prompt):
                console.print(chunk, end="")
                full_text += chunk
        except Exception as e:
            show_error(f"Dialogue failed: {e}")
            return
        console.print()
        console.print()

    def _collect_npc_dialogue(self, npc_name: str, prompt: dict) -> str:
        """Stream NPC dialogue, display it, and return full text."""
        full_text = ""
        console.print()
        console.print(f"[bold yellow]{npc_name}:[/bold yellow]")
        try:
            for chunk in narrate_stream(prompt):
                console.print(chunk, end="")
                full_text += chunk
        except Exception as e:
            show_error(f"Dialogue failed: {e}")
            return ""
        console.print()
        console.print()
        return full_text

    def _process_state_changes(self, response_text: str):
        """Parse and apply state changes from narrator response."""
        changes = parse_state_changes(response_text)
        for change in changes:
            if "move_to" in change:
                new_loc = change["move_to"]
                if new_loc in self.world.locations:
                    self.state.set_location(new_loc)
                    self._discover_current_location()
                    self._show_current_location()
                    self.state.add_journal_entry(
                        f"Traveled to {self.world.locations[new_loc].name}."
                    )
            if "discover" in change:
                disc = change["discover"]
                self.state.discover(disc.get("type", "lore"), disc.get("id", ""))

    def _discover_current_location(self):
        loc_id = self.state.get_location()
        self.state.discover("locations", loc_id)

    def _show_current_location(self):
        loc_id = self.state.get_location()
        location = self.world.locations[loc_id]
        show_location(location, self.world)
```

**Step 2: Commit**

```bash
git add src/rpg_gm/game/engine.py
git commit -m "Add main game engine with exploration and NPC conversation"
```

---

### Task 12: Ingestion CLI

**Files:**
- Modify: `src/rpg_gm/cli.py`

This wires up the full ingestion pipeline with user review.

**Step 1: Implement the ingestion command**

```python
# src/rpg_gm/cli.py
import re
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rpg_gm.ingestion.reader import read_file
from rpg_gm.ingestion.chunker import chunk_text
from rpg_gm.ingestion.embedder import embed_chunks
from rpg_gm.ingestion.extractor import extract_entities_from_chunks
from rpg_gm.world.models import (
    World, Location, NPC, Event, Faction, LoreEntry,
)
from rpg_gm.world.loader import save_world, list_worlds, get_world_dir
from rpg_gm.ui.display import show_entity_for_review, show_info, show_error
from rpg_gm.game.engine import GameEngine

console = Console()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


@click.group()
def main():
    """RPG Game Master — Turn any book into an explorable world."""
    pass


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", prompt="World name", help="Name for this world")
def ingest(file_path: str, name: str):
    """Ingest a PDF or text file into a playable world."""
    world_slug = slugify(name)

    console.print(f"\n[bold]Ingesting[/bold] {file_path} as [cyan]{name}[/cyan] ({world_slug})\n")

    # Stage 1: Read
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Reading file...", total=None)
        pages = read_file(file_path)
        progress.update(task, description=f"Read {len(pages)} pages.")

    # Stage 2: Chunk
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Chunking text...", total=None)
        chunks = chunk_text(pages, chunk_size=500, overlap=50)
        progress.update(task, description=f"Created {len(chunks)} chunks.")

    # Stage 3: Embed
    world_dir = get_world_dir(world_slug)
    chroma_dir = world_dir / "chroma"
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Embedding chunks (this may take a moment on first run)...", total=None)
        embed_chunks(chunks, world_slug, persist_dir=chroma_dir)
        progress.update(task, description=f"Embedded {len(chunks)} chunks into ChromaDB.")

    # Stage 4: Extract entities
    console.print("\n[bold]Extracting world entities with Claude...[/bold]\n")
    world = World(title=name, source_file=file_path)
    existing_names: list[str] = []

    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        console.print(f"[dim]Processing chunks {i+1}-{min(i+batch_size, len(chunks))} of {len(chunks)}...[/dim]")

        try:
            entities = extract_entities_from_chunks(batch, existing_names)
        except Exception as e:
            show_error(f"Extraction failed for batch: {e}")
            continue

        # Review each entity
        quit_extraction = False
        for entity_type, entity_list in entities.items():
            if quit_extraction:
                break
            for entity in entity_list:
                show_entity_for_review(entity_type, entity)
                while True:
                    choice = console.input("[bold]> [/bold]").strip().lower()
                    if choice in ("a", "accept"):
                        _add_entity_to_world(world, entity_type, entity)
                        existing_names.append(entity.get("name", ""))
                        show_info("Accepted.")
                        break
                    elif choice in ("s", "skip"):
                        show_info("Skipped.")
                        break
                    elif choice in ("q", "quit"):
                        quit_extraction = True
                        break
                    else:
                        console.print("[dim]Press a (accept), s (skip), or q (quit)[/dim]")

        if quit_extraction:
            console.print("[dim]Stopped extraction early.[/dim]")
            break

    # Save world
    save_world(world, world_slug)
    total = (
        len(world.locations) + len(world.npcs) + len(world.events)
        + len(world.factions) + len(world.lore)
    )
    console.print(
        f"\n[bold green]World saved![/bold green] "
        f"{total} entities in [cyan]worlds/{world_slug}/[/cyan]"
    )
    console.print(f"Run [bold]rpg-gm play {world_slug}[/bold] to explore.\n")


def _add_entity_to_world(world: World, entity_type: str, entity: dict):
    """Add a parsed entity dict to the world."""
    name = entity.get("name") or entity.get("title", "unknown")
    entity_id = slugify(name)

    if entity_type == "locations":
        world.locations[entity_id] = Location(
            id=entity_id,
            name=name,
            description=entity.get("description", ""),
            connections=[slugify(c) for c in entity.get("connections", [])],
            details=entity.get("details", []),
            npcs=[],
            source_pages=entity.get("source_pages", []),
        )
    elif entity_type == "npcs":
        loc_id = slugify(entity.get("location", "unknown"))
        npc = NPC(
            id=entity_id,
            name=name,
            role=entity.get("role", ""),
            description=entity.get("description", ""),
            personality=entity.get("personality", ""),
            knowledge=entity.get("knowledge", []),
            location_id=loc_id,
            relationships={
                slugify(k): v
                for k, v in entity.get("relationships", {}).items()
            },
            source_pages=entity.get("source_pages", []),
        )
        world.npcs[entity_id] = npc
        # Add NPC to their location if it exists
        if loc_id in world.locations:
            if entity_id not in world.locations[loc_id].npcs:
                world.locations[loc_id].npcs.append(entity_id)
    elif entity_type == "events":
        world.events[entity_id] = Event(
            id=entity_id,
            name=name,
            description=entity.get("description", ""),
            time_period=entity.get("time_period", ""),
            participants=[slugify(p) for p in entity.get("participants", [])],
            locations=[slugify(loc) for loc in entity.get("locations", [])],
            source_pages=entity.get("source_pages", []),
        )
    elif entity_type == "factions":
        world.factions[entity_id] = Faction(
            id=entity_id,
            name=name,
            description=entity.get("description", ""),
            members=[slugify(m) for m in entity.get("members", [])],
            goals=entity.get("goals", ""),
            source_pages=entity.get("source_pages", []),
        )
    elif entity_type == "lore":
        world.lore[entity_id] = LoreEntry(
            id=entity_id,
            category=entity.get("category", "other"),
            title=name,
            content=entity.get("content", ""),
            related_entities=[slugify(r) for r in entity.get("related_entities", [])],
            source_pages=entity.get("source_pages", []),
        )


@main.command()
@click.argument("world_name", required=False)
def play(world_name: str | None):
    """Play an ingested world."""
    if not world_name:
        worlds = list_worlds()
        if not worlds:
            show_error("No worlds found. Run 'rpg-gm ingest <file>' first.")
            return
        console.print("[bold]Available worlds:[/bold]")
        for w in worlds:
            console.print(f"  [cyan]{w}[/cyan]")
        return

    try:
        engine = GameEngine(world_name)
        engine.run()
    except FileNotFoundError:
        show_error(f"World '{world_name}' not found. Run 'rpg-gm play' to see available worlds.")
    except KeyboardInterrupt:
        show_info("\nGame interrupted. Progress was auto-saved.")
```

**Step 2: Commit**

```bash
git add src/rpg_gm/cli.py
git commit -m "Wire up full ingestion and play CLI commands"
```

---

### Task 13: Integration Test — End to End

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/fixtures/sample.txt`

**Step 1: Create a sample text file for testing**

```text
# tests/fixtures/sample.txt
The Agora of Athens was the central public space in ancient Athens. It served as the city's marketplace and civic center. Merchants sold pottery, olive oil, and textiles from wooden stalls. Philosophers gathered beneath the painted colonnades known as stoas to debate ideas about justice, beauty, and the good life.

Socrates was a philosopher who spent most of his days in the Agora, engaging anyone who would listen in conversation. He was known for his method of questioning — the Socratic method — where he would ask probing questions to expose contradictions in people's beliefs. He wore a simple cloak and went barefoot. His wife Xanthippe was known for her sharp temper.

The Acropolis rose above the city on a rocky outcrop. At its summit stood the Parthenon, a temple dedicated to Athena, the patron goddess of Athens. The Parthenon was built between 447 and 432 BC under the direction of Pericles. Its marble columns gleamed white in the Mediterranean sun. Inside stood a massive gold and ivory statue of Athena created by the sculptor Phidias.

The road from the Agora led uphill to the Acropolis through the Propylaea, a monumental gateway. Along the way, vendors sold small clay figurines of the gods as offerings. Citizens climbed this path during the Panathenaic festival, a grand procession honoring Athena held every four years.

Pericles was a prominent statesman and orator who led Athens during its golden age in the 5th century BC. He championed democracy, funded the construction of the Parthenon, and delivered the famous Funeral Oration honoring fallen Athenian soldiers. He was known for his calm demeanor and persuasive speech.
```

**Step 2: Write integration test**

```python
# tests/test_integration.py
"""Integration test: ingest a sample file, verify world is created correctly."""
from pathlib import Path

from rpg_gm.ingestion.reader import read_file
from rpg_gm.ingestion.chunker import chunk_text
from rpg_gm.ingestion.embedder import embed_chunks, query_chunks
from rpg_gm.world.models import World, Location, NPC
from rpg_gm.world.loader import save_world, load_world


def test_full_pipeline_without_claude(tmp_path):
    """Test ingestion pipeline end-to-end, skipping the Claude extraction step."""
    sample_file = Path(__file__).parent / "fixtures" / "sample.txt"

    # Read
    pages = read_file(str(sample_file))
    assert len(pages) == 1

    # Chunk
    chunks = chunk_text(pages, chunk_size=300, overlap=50)
    assert len(chunks) >= 3

    # Embed
    chroma_dir = tmp_path / "chroma"
    embed_chunks(chunks, "test-athens", persist_dir=chroma_dir)

    # Query
    results = query_chunks("Socrates philosophy", "test-athens", persist_dir=chroma_dir, n_results=2)
    assert len(results) == 2
    assert any("Socrates" in r["text"] for r in results)

    # Build world manually (simulating Claude extraction output)
    world = World(
        title="Ancient Athens",
        source_file=str(sample_file),
        locations={
            "agora": Location(
                id="agora",
                name="The Agora of Athens",
                description="The central marketplace and civic center of Athens.",
                connections=["acropolis"],
                details=["painted colonnades", "merchant stalls"],
                npcs=["socrates"],
                source_pages=[1],
            ),
            "acropolis": Location(
                id="acropolis",
                name="The Acropolis",
                description="A rocky outcrop above the city, crowned by the Parthenon.",
                connections=["agora"],
                details=["Parthenon", "Propylaea gateway"],
                npcs=["pericles"],
                source_pages=[1],
            ),
        },
        npcs={
            "socrates": NPC(
                id="socrates",
                name="Socrates",
                role="philosopher",
                description="A philosopher who debates in the Agora.",
                personality="Inquisitive, ironic, persistent questioner.",
                knowledge=["philosophy", "ethics", "Socratic method"],
                location_id="agora",
                relationships={"xanthippe": "wife"},
                source_pages=[1],
            ),
            "pericles": NPC(
                id="pericles",
                name="Pericles",
                role="statesman",
                description="Led Athens during its golden age.",
                personality="Calm, persuasive, democratic idealist.",
                knowledge=["politics", "democracy", "Athenian history"],
                location_id="acropolis",
                relationships={},
                source_pages=[1],
            ),
        },
    )

    # Save and reload
    save_world(world, "test-athens", base_dir=tmp_path)
    loaded = load_world("test-athens", base_dir=tmp_path)
    assert loaded.title == "Ancient Athens"
    assert len(loaded.locations) == 2
    assert len(loaded.npcs) == 2
    assert "socrates" in loaded.locations["agora"].npcs
```

**Step 3: Run integration test**

```bash
uv run pytest tests/test_integration.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_integration.py tests/fixtures/sample.txt
git commit -m "Add integration test with sample fixture"
```

---

### Task 14: .gitignore and Final Cleanup

**Files:**
- Create: `.gitignore`

**Step 1: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environment
.venv/

# World data (generated, potentially large)
worlds/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

**Step 2: Verify everything works**

```bash
uv run rpg-gm --help
uv run pytest -v
```

Expected: CLI help shows `ingest` and `play` commands. All tests pass.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "Add .gitignore"
```

---

## Execution Summary

| Task | Description | Est. Complexity |
|------|-------------|-----------------|
| 1 | Project scaffolding | Low |
| 2 | World data models | Low |
| 3 | World loader (JSON) | Low |
| 4 | PDF reader + chunker | Medium |
| 5 | ChromaDB embedder | Medium |
| 6 | Claude entity extractor | Medium |
| 7 | Rich terminal UI | Medium |
| 8 | SQLite game state | Medium |
| 9 | Claude narrator (RAG) | Medium |
| 10 | Game commands | Low |
| 11 | Game engine (main loop) | High |
| 12 | Ingestion CLI | High |
| 13 | Integration test | Low |
| 14 | .gitignore + cleanup | Low |

**Total: 14 tasks.** Tasks 1-6 are independent enough to build in parallel. Tasks 7-12 have sequential dependencies. Tasks 13-14 are final verification.
