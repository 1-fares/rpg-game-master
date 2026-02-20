"""Tests for world JSON save/load/list."""

from pathlib import Path

from rpg_gm.world.loader import get_world_dir, list_worlds, load_world, save_world
from rpg_gm.world.models import Location, NPC, World


def _make_world() -> World:
    return World(
        title="Test Realm",
        source_file="test.pdf",
        locations={
            "tavern": Location(
                id="tavern",
                name="The Rusty Flagon",
                description="A dimly lit tavern.",
                npcs=["barkeep"],
            )
        },
        npcs={
            "barkeep": NPC(
                id="barkeep",
                name="Grul",
                role="Bartender",
                description="A gruff half-orc.",
                personality="Grumpy but fair.",
                location_id="tavern",
            )
        },
    )


def test_get_world_dir_default():
    from rpg_gm.world.loader import DEFAULT_WORLDS_DIR
    result = get_world_dir("my_world")
    assert result == DEFAULT_WORLDS_DIR / "my_world"
    assert result.is_absolute()


def test_get_world_dir_custom_base(tmp_path: Path):
    result = get_world_dir("my_world", base_dir=tmp_path)
    assert result == tmp_path / "my_world"


def test_save_and_load_world(tmp_path: Path):
    world = _make_world()
    path = save_world(world, "test_realm", base_dir=tmp_path)

    assert path.exists()
    assert path.name == "world.json"

    loaded = load_world("test_realm", base_dir=tmp_path)
    assert loaded == world
    assert loaded.title == "Test Realm"
    assert "tavern" in loaded.locations
    assert loaded.npcs["barkeep"].name == "Grul"


def test_load_world_not_found(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError, match="World 'nope' not found"):
        load_world("nope", base_dir=tmp_path)


def test_list_worlds_empty(tmp_path: Path):
    assert list_worlds(base_dir=tmp_path) == []


def test_list_worlds_nonexistent_base():
    assert list_worlds(base_dir=Path("/tmp/does_not_exist_xyz")) == []


def test_list_worlds(tmp_path: Path):
    for name in ["beta", "alpha", "gamma"]:
        save_world(_make_world(), name, base_dir=tmp_path)

    result = list_worlds(base_dir=tmp_path)
    assert result == ["alpha", "beta", "gamma"]  # sorted


def test_list_worlds_ignores_dirs_without_world_json(tmp_path: Path):
    (tmp_path / "empty_dir").mkdir()
    save_world(_make_world(), "valid", base_dir=tmp_path)

    result = list_worlds(base_dir=tmp_path)
    assert result == ["valid"]
