"""Command parser and entity name matching utilities."""

from dataclasses import dataclass


@dataclass
class ParsedCommand:
    name: str
    args: str = ""


def parse_input(raw: str) -> ParsedCommand | None:
    """Parse player input. Returns ParsedCommand for slash commands, None for free text."""
    raw = raw.strip()
    if not raw.startswith("/"):
        return None
    parts = raw[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return ParsedCommand(name=name, args=args)


def find_npc_by_name(name: str, npc_ids: list[str], world) -> str | None:
    """Match an NPC name to an ID. Returns NPC ID or None.

    Tries exact match first, then substring. If multiple substring matches,
    returns None (ambiguous).
    """
    name_lower = name.lower()

    # Exact match on name
    for npc_id in npc_ids:
        npc = world.npcs.get(npc_id)
        if npc and npc.name.lower() == name_lower:
            return npc_id

    # Substring match
    matches = []
    for npc_id in npc_ids:
        npc = world.npcs.get(npc_id)
        if npc and name_lower in npc.name.lower():
            matches.append(npc_id)

    if len(matches) == 1:
        return matches[0]
    return None  # No match or ambiguous


def find_location_by_name(name: str, location_ids: list[str], world) -> str | None:
    """Match a location name to an ID. Returns location ID or None.

    Tries exact match first, then substring, then ID match.
    """
    name_lower = name.lower()

    # Exact match on name
    for loc_id in location_ids:
        loc = world.locations.get(loc_id)
        if loc and loc.name.lower() == name_lower:
            return loc_id

    # Substring match on name
    matches = []
    for loc_id in location_ids:
        loc = world.locations.get(loc_id)
        if loc and name_lower in loc.name.lower():
            matches.append(loc_id)
    if len(matches) == 1:
        return matches[0]

    # ID match
    for loc_id in location_ids:
        if name_lower in loc_id.lower():
            return loc_id

    return None
