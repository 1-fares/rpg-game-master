from rpg_gm.game.narrator import build_gm_prompt, build_npc_prompt, NARRATOR_TOOLS, NPC_TOOLS
from rpg_gm.world.models import Location, NPC, World


def _make_world() -> World:
    return World(
        title="Ancient Athens",
        source_file="athens.pdf",
        locations={
            "agora": Location(id="agora", name="The Agora", description="The central marketplace of Athens.", connections=["acropolis"], details=["marble columns"], npcs=["socrates"], source_pages=[1]),
            "acropolis": Location(id="acropolis", name="The Acropolis", description="A rocky hill.", connections=["agora"], details=[], npcs=[], source_pages=[1]),
        },
        npcs={
            "socrates": NPC(id="socrates", name="Socrates", role="philosopher", description="A famous Athenian philosopher.", personality="Inquisitive, ironic, humble.", knowledge=["philosophy", "ethics"], location_id="agora", relationships={}, source_pages=[1]),
        },
    )


def test_build_gm_prompt():
    world = _make_world()
    passages = [{"text": "The Agora was bustling with merchants.", "page": 5}]
    history = [{"role": "user", "content": "I look around."}]
    prompt = build_gm_prompt(world=world, location_id="agora", passages=passages, history=history, player_input="I walk toward the columns.")
    assert "Ancient Athens" in prompt["system"]
    assert "The Agora" in prompt["system"]
    assert "Socrates" in prompt["system"]
    assert "acropolis" in prompt["system"]  # Available locations
    assert "bustling with merchants" in prompt["messages"][-1]["content"]
    assert "walk toward the columns" in prompt["messages"][-1]["content"]
    assert prompt["tools"] == NARRATOR_TOOLS


def test_build_gm_prompt_includes_location_ids():
    """Verify available locations include IDs for move_player tool."""
    world = _make_world()
    prompt = build_gm_prompt(world=world, location_id="agora", passages=[], history=[], player_input="test")
    assert "(agora)" in prompt["system"]
    assert "(acropolis)" in prompt["system"]


def test_build_npc_prompt():
    world = _make_world()
    npc = world.npcs["socrates"]
    passages = [{"text": "Socrates believed virtue was knowledge.", "page": 10}]
    prompt = build_npc_prompt(npc=npc, passages=passages, history=[], player_input="What is virtue?")
    assert "Socrates" in prompt["system"]
    assert "Inquisitive" in prompt["system"]
    assert "philosophy" in prompt["system"]
    assert "virtue was knowledge" in prompt["messages"][-1]["content"]
    assert prompt["tools"] == NPC_TOOLS


def test_narrator_tools_have_move_and_discover():
    tool_names = [t["name"] for t in NARRATOR_TOOLS]
    assert "move_player" in tool_names
    assert "discover_entity" in tool_names


def test_npc_tools_have_end_conversation():
    tool_names = [t["name"] for t in NPC_TOOLS]
    assert "end_conversation" in tool_names
    assert "discover_entity" in tool_names
