from rpg_gm.ingestion.extractor import (
    build_extraction_prompt,
    parse_extraction_response,
    EXTRACTION_TOOLS,
)
from rpg_gm.ingestion.chunker import Chunk


def test_build_extraction_prompt():
    chunks = [
        Chunk(index=0, text="The Agora was the central marketplace of Athens.", page=1),
        Chunk(index=1, text="Socrates often debated in the Agora.", page=2),
    ]
    prompt = build_extraction_prompt(chunks, existing_names=[])
    assert "Agora" in prompt
    assert "Socrates" in prompt
    assert "report_extracted_entities" in prompt


def test_build_extraction_prompt_includes_existing():
    chunks = [Chunk(index=0, text="Some text.", page=1)]
    prompt = build_extraction_prompt(chunks, existing_names=["The Agora", "Socrates"])
    assert "The Agora" in prompt
    assert "Socrates" in prompt
    assert "Already extracted" in prompt


def test_parse_extraction_response_valid():
    raw = {
        "locations": [{"name": "The Agora", "description": "Central marketplace.", "connections": [], "details": ["stalls"], "source_pages": [1]}],
        "npcs": [{"name": "Socrates", "role": "philosopher", "description": "A philosopher.", "personality": "Inquisitive.", "knowledge": ["philosophy"], "location": "The Agora", "relationships": {}, "source_pages": [2]}],
        "events": [],
        "factions": [],
        "lore": [],
    }
    entities = parse_extraction_response(raw)
    assert len(entities["locations"]) == 1
    assert len(entities["npcs"]) == 1
    assert entities["locations"][0]["name"] == "The Agora"


def test_parse_extraction_response_empty():
    raw = {"locations": [], "npcs": [], "events": [], "factions": [], "lore": []}
    entities = parse_extraction_response(raw)
    assert all(len(v) == 0 for v in entities.values())


def test_parse_extraction_response_missing_keys():
    """Missing keys should default to empty lists."""
    raw = {"locations": [{"name": "Test", "description": "Desc"}]}
    entities = parse_extraction_response(raw)
    assert len(entities["locations"]) == 1
    assert entities["npcs"] == []


def test_extraction_tools_schema():
    """Verify the tool schema is well-formed."""
    assert len(EXTRACTION_TOOLS) == 1
    tool = EXTRACTION_TOOLS[0]
    assert tool["name"] == "report_extracted_entities"
    schema = tool["input_schema"]
    assert "locations" in schema["properties"]
    assert "npcs" in schema["properties"]
    assert "events" in schema["properties"]
    assert "factions" in schema["properties"]
    assert "lore" in schema["properties"]
