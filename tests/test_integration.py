"""Integration test: ingest a sample file, verify world is created correctly."""
from pathlib import Path

from rpg_gm.ingestion.reader import read_file
from rpg_gm.ingestion.chunker import chunk_text
from rpg_gm.ingestion.embedder import embed_chunks, query_chunks
from rpg_gm.world.models import World, Location, NPC
from rpg_gm.world.loader import save_world, load_world
from rpg_gm.game.state import GameState
from rpg_gm.game.commands import parse_input, find_npc_by_name


def test_full_pipeline_without_claude(tmp_path):
    """Test ingestion pipeline end-to-end, skipping the Claude extraction step."""
    sample_file = Path(__file__).parent / "fixtures" / "sample.txt"

    # Read
    pages = read_file(str(sample_file))
    assert len(pages) == 1

    # Chunk (using revised 1200 char / 150 overlap settings)
    chunks = chunk_text(pages, chunk_size=1200, overlap=150)
    assert len(chunks) >= 1

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


def test_game_state_with_world(tmp_path):
    """Test game state works with a loaded world."""
    db_path = tmp_path / "test.db"
    state = GameState(str(db_path))
    state.init("test-athens", "agora")

    assert state.get_location() == "agora"
    state.set_location("acropolis")
    assert state.get_location() == "acropolis"
    assert state.get_visited() == {"agora", "acropolis"}

    state.discover("locations", "agora")
    state.discover("npcs", "socrates")
    discovered = state.get_discovered()
    assert "agora" in discovered["locations"]
    assert "socrates" in discovered["npcs"]

    state.add_journal_entry("Arrived at the Agora.")
    assert len(state.get_journal()) == 1

    state.close()


def test_command_parsing_with_world():
    """Test command parser with actual world data."""
    world = World(
        title="Test",
        source_file="test.txt",
        locations={
            "agora": Location(id="agora", name="The Agora", description="Market.", connections=[], details=[], npcs=["socrates"], source_pages=[]),
        },
        npcs={
            "socrates": NPC(id="socrates", name="Socrates", role="philosopher", description="", personality="", knowledge=[], location_id="agora", relationships={}, source_pages=[]),
        },
    )

    cmd = parse_input("/talk Socrates")
    assert cmd is not None
    assert cmd.name == "talk"
    assert cmd.args == "Socrates"

    npc_id = find_npc_by_name("Socrates", ["socrates"], world)
    assert npc_id == "socrates"
