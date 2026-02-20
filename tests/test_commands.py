from rpg_gm.game.commands import parse_input, find_npc_by_name, find_location_by_name, ParsedCommand
from rpg_gm.world.models import World, Location, NPC


def test_parse_slash_command():
    cmd = parse_input("/look")
    assert cmd is not None
    assert cmd.name == "look"
    assert cmd.args == ""


def test_parse_slash_command_with_args():
    cmd = parse_input("/talk Socrates")
    assert cmd.name == "talk"
    assert cmd.args == "Socrates"


def test_parse_free_text_returns_none():
    assert parse_input("walk to the temple") is None


def test_parse_empty_returns_none():
    assert parse_input("") is None


def test_parse_case_insensitive():
    cmd = parse_input("/LOOK")
    assert cmd.name == "look"


def _make_world():
    return World(
        title="Test",
        source_file="test.txt",
        locations={
            "agora": Location(id="agora", name="The Agora", description="Market.", connections=["acropolis"], details=[], npcs=["socrates", "plato"], source_pages=[]),
            "acropolis": Location(id="acropolis", name="The Acropolis", description="Hill.", connections=["agora"], details=[], npcs=[], source_pages=[]),
        },
        npcs={
            "socrates": NPC(id="socrates", name="Socrates", role="philosopher", description="", personality="", knowledge=[], location_id="agora", relationships={}, source_pages=[]),
            "plato": NPC(id="plato", name="Plato", role="philosopher", description="", personality="", knowledge=[], location_id="agora", relationships={}, source_pages=[]),
        },
    )


def test_find_npc_exact_match():
    world = _make_world()
    assert find_npc_by_name("Socrates", ["socrates", "plato"], world) == "socrates"


def test_find_npc_case_insensitive():
    world = _make_world()
    assert find_npc_by_name("socrates", ["socrates", "plato"], world) == "socrates"


def test_find_npc_substring():
    world = _make_world()
    assert find_npc_by_name("soc", ["socrates", "plato"], world) == "socrates"


def test_find_npc_no_match():
    world = _make_world()
    assert find_npc_by_name("aristotle", ["socrates", "plato"], world) is None


def test_find_npc_ambiguous_returns_none():
    """Two philosophers both have 'o' in their name -- should return None."""
    world = _make_world()
    assert find_npc_by_name("o", ["socrates", "plato"], world) is None


def test_find_location_exact():
    world = _make_world()
    assert find_location_by_name("The Agora", ["agora", "acropolis"], world) == "agora"


def test_find_location_substring():
    world = _make_world()
    assert find_location_by_name("agora", ["agora", "acropolis"], world) == "agora"


def test_find_location_by_id():
    world = _make_world()
    assert find_location_by_name("acropolis", ["agora", "acropolis"], world) == "acropolis"
