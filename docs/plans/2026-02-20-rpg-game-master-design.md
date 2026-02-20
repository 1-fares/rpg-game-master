# RPG Game Master — Design Document

**Date:** 2026-02-20
**Status:** Approved

## Overview

A terminal-based RPG game master that turns any book or PDF into an explorable, interactive world. No combat — the focus is on exploration, discovery, conversation, and learning about the world described in the source material. Runs locally on WSL (Ubuntu, no GPU).

## Key Decisions

- **Source material**: Non-fiction/history (primary use case)
- **AI backend**: Claude API only (no local LLM)
- **Extraction**: AI-assisted with user review/approval per entity
- **UI**: Rich library (panels, color, tables — not a full TUI)
- **Architecture**: Two-phase (separate ingestion CLI and game loop)

## Architecture

### Two-Phase Design

**Phase 1 — Ingestion** (`rpg-gm ingest <file>`): Reads a PDF/text file, chunks it, embeds into ChromaDB, uses Claude to extract world entities. User reviews each extraction (accept/skip). Produces a world directory with JSON + ChromaDB data.

**Phase 2 — Game Loop** (`rpg-gm play <world>`): Loads world data and ChromaDB. Player navigates locations, talks to NPCs, examines details. Each interaction uses RAG retrieval + Claude for narration. Game state saved to SQLite.

## Project Structure

```
rpg-game-master/
├── pyproject.toml
├── src/
│   └── rpg_gm/
│       ├── __init__.py
│       ├── cli.py              # Entry points: ingest, play
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── reader.py       # PDF/text file reading
│       │   ├── chunker.py      # Text chunking logic
│       │   ├── embedder.py     # ChromaDB embedding
│       │   └── extractor.py    # Claude-based entity extraction
│       ├── world/
│       │   ├── __init__.py
│       │   ├── models.py       # Pydantic models
│       │   └── loader.py       # Load/save world JSON files
│       ├── game/
│       │   ├── __init__.py
│       │   ├── engine.py       # Main game loop
│       │   ├── commands.py     # Slash command handlers
│       │   ├── narrator.py     # Claude narration with RAG
│       │   └── state.py        # SQLite game state management
│       └── ui/
│           ├── __init__.py
│           └── display.py      # Rich-based terminal output
├── worlds/                     # Generated world data (per-book)
│   └── <book-slug>/
│       ├── world.json
│       ├── chroma/
│       └── saves/
└── docs/
    └── plans/
```

## Dependencies

- `anthropic` — Claude API
- `chromadb` — vector store
- `sentence-transformers` — embedding model (all-MiniLM-L6-v2, ~80MB, CPU)
- `pymupdf` — PDF reading
- `rich` — terminal UI
- `pydantic` — data models
- `click` — CLI framework

## Data Models

### Location
- id, name, description
- connections (list of location IDs)
- details (examinable things)
- npcs (NPC IDs present here)
- source_pages

### NPC
- id, name, role, description
- personality (brief guide for Claude)
- knowledge (topics they can discuss)
- location_id
- relationships (NPC ID -> description)
- source_pages

### Event
- id, name, description
- time_period
- participants (NPC IDs)
- locations (location IDs)
- source_pages

### Faction
- id, name, description
- members (NPC IDs), goals
- source_pages

### LoreEntry
- id, category, title, content
- related_entities
- source_pages

### World (container)
- title, source_file
- locations, npcs, events, factions, lore (all as dicts keyed by ID)

## Ingestion Pipeline

1. **Read & Chunk**: PyMuPDF extracts text, split into ~500-token chunks with ~50-token overlap. Each chunk gets a sequential ID and page number.
2. **Embed**: Chunks embedded via all-MiniLM-L6-v2, stored in persistent ChromaDB collection. Metadata: page number, chunk index, preview text.
3. **Extract Entities**: Process chunks in batches of 5. Claude extracts locations/NPCs/events/factions/lore as structured JSON. Each entity presented to user for accept/skip. Duplicates checked against existing entities.

## Game Loop

### State (SQLite)
- player_location, journal_entries, discovered_entities, conversation_history (ring buffer ~20 messages), visited_locations

### Main Loop
1. Display current location (Rich panel)
2. Wait for player input
3. Slash command → handle directly; free text → Claude narration with RAG
4. Update game state
5. Loop

### Slash Commands

| Command | Action |
|---------|--------|
| `/look` | Describe current location in detail (RAG-enhanced) |
| `/map` | Show visited locations and connections |
| `/talk <npc>` | Start conversation with an NPC |
| `/examine <thing>` | Examine a detail or object |
| `/journal` | Show journal entries |
| `/discoveries` | Show discovery tracker with completion % |
| `/save` | Save game state |
| `/load` | Load a saved game |
| `/help` | Show available commands |
| `/quit` | Exit |

### RAG Flow (per Claude call)
1. Query ChromaDB for top 5 relevant chunks using player input + context
2. Assemble prompt: system instructions, world context, retrieved passages, recent history, player input
3. Stream Claude's response
4. Parse inline JSON tags for state changes (move_to, discover)

## AI Prompt Design

### Game Master System Prompt
- Second person present tense narration
- Faithful to source material, never invents unsupported facts
- 2-4 paragraphs per response, sensory detail without verbosity
- Ends with exploration hooks
- Includes structured JSON tags for state changes

### NPC Conversation Prompt
- Stays in character with NPC's voice, vocabulary, worldview
- Constrained to NPC's knowledge topics
- References source passages
- 1-3 paragraphs of natural dialogue

### Prompt Assembly
System prompt → world context (location, NPCs) → retrieved passages (top 5) → recent history (~10 exchanges) → player input

## MVP Scope

### MVP (first build)
- PDF reading + text chunking
- ChromaDB embedding
- Claude entity extraction with accept/skip
- Save world.json
- Load world, display starting location
- Free-text input → Claude narration with RAG
- `/look`, `/talk <npc>`, `/map` (simple list), `/quit`
- Basic SQLite state (location, visited places)
- Streaming Claude responses
- Rich panels for locations, colored NPC dialogue

### Layer 2
- `/examine` with detail lookups
- `/journal` with auto-journaling
- `/discoveries` tracker with completion %
- `/save` and `/load` with multiple save slots
- Entity editing during ingestion
- Conversation history persistence
- Duplicate entity merging

### Layer 3
- ASCII map visualization
- NPC relationship web
- Categorized journal entries
- Guided tour mode
- Multi-book support
- Export discoveries as document
