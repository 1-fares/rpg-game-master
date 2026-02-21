import os
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
@click.option("--review", is_flag=True, help="Manually review each entity before accepting")
def ingest(file_path: str, name: str, review: bool):
    """Ingest a PDF or text file into a playable world."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        show_error("ANTHROPIC_API_KEY not set. Export it before running ingest.")
        raise SystemExit(1)

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
        chunks = chunk_text(pages, chunk_size=1200, overlap=150)
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

    batch_size = 10  # Revised from 5 to 10 per external review
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
                if review:
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
                else:
                    _add_entity_to_world(world, entity_type, entity)
                    existing_names.append(entity.get("name", ""))

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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        show_error("ANTHROPIC_API_KEY not set. Export it before running play.")
        return

    try:
        engine = GameEngine(world_name)
        engine.run()
    except FileNotFoundError:
        show_error(f"World '{world_name}' not found. Run 'rpg-gm play' to see available worlds.")
    except KeyboardInterrupt:
        show_info("\nGame interrupted. Progress was auto-saved.")
