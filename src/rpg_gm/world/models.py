"""Pydantic models for world entities."""

from pydantic import BaseModel


class Location(BaseModel):
    id: str
    name: str
    description: str
    connections: list[str] = []  # IDs of connected locations
    details: list[str] = []  # Examinable details
    npcs: list[str] = []  # NPC IDs present here
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
