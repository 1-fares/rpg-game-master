"""Save, load, and list worlds as JSON files."""

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
