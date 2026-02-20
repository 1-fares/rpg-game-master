"""Tests for world data models."""

import json

from rpg_gm.world.models import (
    Event,
    Faction,
    Location,
    LoreEntry,
    NPC,
    World,
)


# --- Location ---


class TestLocation:
    def test_create_with_all_fields(self):
        loc = Location(
            id="tavern",
            name="The Rusty Flagon",
            description="A dimly lit tavern smelling of ale.",
            connections=["market", "alley"],
            details=["A notice board hangs on the wall.", "The bar is sticky."],
            npcs=["bartender"],
            source_pages=[12, 13],
        )
        assert loc.id == "tavern"
        assert loc.name == "The Rusty Flagon"
        assert loc.connections == ["market", "alley"]
        assert loc.details == ["A notice board hangs on the wall.", "The bar is sticky."]
        assert loc.npcs == ["bartender"]
        assert loc.source_pages == [12, 13]

    def test_defaults_are_empty_lists(self):
        loc = Location(id="void", name="Void", description="Nothing here.")
        assert loc.connections == []
        assert loc.details == []
        assert loc.npcs == []
        assert loc.source_pages == []


# --- NPC ---


class TestNPC:
    def test_create_with_all_fields(self):
        npc = NPC(
            id="bartender",
            name="Grog",
            role="Tavern keeper",
            description="A burly half-orc.",
            personality="Gruff but fair.",
            knowledge=["Knows about the missing shipment."],
            location_id="tavern",
            relationships={"mayor": "distrusts"},
            source_pages=[14],
        )
        assert npc.id == "bartender"
        assert npc.name == "Grog"
        assert npc.role == "Tavern keeper"
        assert npc.personality == "Gruff but fair."
        assert npc.knowledge == ["Knows about the missing shipment."]
        assert npc.location_id == "tavern"
        assert npc.relationships == {"mayor": "distrusts"}
        assert npc.source_pages == [14]

    def test_defaults_are_empty(self):
        npc = NPC(
            id="guard",
            name="Guard",
            role="Guard",
            description="A town guard.",
            personality="Stoic.",
            location_id="gate",
        )
        assert npc.knowledge == []
        assert npc.relationships == {}
        assert npc.source_pages == []


# --- Event ---


class TestEvent:
    def test_create_with_all_fields(self):
        event = Event(
            id="festival",
            name="Harvest Festival",
            description="Annual celebration of the harvest.",
            time_period="Autumn, Year 342",
            participants=["mayor", "bartender"],
            locations=["market", "tavern"],
            source_pages=[20, 21],
        )
        assert event.id == "festival"
        assert event.name == "Harvest Festival"
        assert event.time_period == "Autumn, Year 342"
        assert event.participants == ["mayor", "bartender"]
        assert event.locations == ["market", "tavern"]
        assert event.source_pages == [20, 21]

    def test_defaults_are_empty_lists(self):
        event = Event(
            id="fire",
            name="Great Fire",
            description="A devastating fire.",
            time_period="Summer, Year 300",
        )
        assert event.participants == []
        assert event.locations == []
        assert event.source_pages == []


# --- Faction ---


class TestFaction:
    def test_create_with_all_fields(self):
        faction = Faction(
            id="thieves_guild",
            name="Shadow Hand",
            description="An underground network of thieves.",
            members=["rogue_leader", "pickpocket"],
            goals="Control the black market.",
            source_pages=[30, 31, 32],
        )
        assert faction.id == "thieves_guild"
        assert faction.name == "Shadow Hand"
        assert faction.members == ["rogue_leader", "pickpocket"]
        assert faction.goals == "Control the black market."
        assert faction.source_pages == [30, 31, 32]

    def test_defaults_are_empty_lists(self):
        faction = Faction(
            id="guild",
            name="Guild",
            description="A guild.",
            goals="Profit.",
        )
        assert faction.members == []
        assert faction.source_pages == []


# --- LoreEntry ---


class TestLoreEntry:
    def test_create_with_all_fields(self):
        lore = LoreEntry(
            id="creation_myth",
            category="mythology",
            title="The Creation of the World",
            content="In the beginning there was nothing...",
            related_entities=["elder_god", "first_city"],
            source_pages=[1, 2, 3],
        )
        assert lore.id == "creation_myth"
        assert lore.category == "mythology"
        assert lore.title == "The Creation of the World"
        assert lore.content == "In the beginning there was nothing..."
        assert lore.related_entities == ["elder_god", "first_city"]
        assert lore.source_pages == [1, 2, 3]

    def test_defaults_are_empty_lists(self):
        lore = LoreEntry(
            id="note",
            category="misc",
            title="A Note",
            content="Something.",
        )
        assert lore.related_entities == []
        assert lore.source_pages == []


# --- World ---


class TestWorld:
    def _make_world(self) -> World:
        """Build a small but complete World for testing."""
        loc = Location(
            id="tavern",
            name="The Rusty Flagon",
            description="A tavern.",
            connections=["market"],
            npcs=["bartender"],
        )
        npc = NPC(
            id="bartender",
            name="Grog",
            role="Tavern keeper",
            description="A burly half-orc.",
            personality="Gruff.",
            location_id="tavern",
        )
        event = Event(
            id="festival",
            name="Harvest Festival",
            description="Annual celebration.",
            time_period="Autumn",
        )
        faction = Faction(
            id="guild",
            name="Merchants Guild",
            description="Trade union.",
            goals="Profit.",
        )
        lore = LoreEntry(
            id="myth",
            category="mythology",
            title="Creation",
            content="In the beginning...",
        )
        return World(
            title="Test World",
            source_file="test.pdf",
            locations={"tavern": loc},
            npcs={"bartender": npc},
            events={"festival": event},
            factions={"guild": faction},
            lore={"myth": lore},
        )

    def test_create_with_entities(self):
        world = self._make_world()
        assert world.title == "Test World"
        assert world.source_file == "test.pdf"
        assert "tavern" in world.locations
        assert "bartender" in world.npcs
        assert "festival" in world.events
        assert "guild" in world.factions
        assert "myth" in world.lore

    def test_defaults_are_empty_dicts(self):
        world = World(title="Empty World", source_file="empty.pdf")
        assert world.locations == {}
        assert world.npcs == {}
        assert world.events == {}
        assert world.factions == {}
        assert world.lore == {}

    def test_serialization_roundtrip(self):
        """model_dump_json -> model_validate_json produces an equal World."""
        original = self._make_world()
        json_str = original.model_dump_json()
        restored = World.model_validate_json(json_str)
        assert restored == original

    def test_serialization_roundtrip_via_dict(self):
        """model_dump -> json.loads/dumps -> model_validate produces an equal World."""
        original = self._make_world()
        data = original.model_dump()
        json_str = json.dumps(data)
        restored = World.model_validate(json.loads(json_str))
        assert restored == original
