import anthropic

from rpg_gm.ingestion.chunker import Chunk

# Define extraction tools for Claude's tool_use
EXTRACTION_TOOLS = [
    {
        "name": "report_extracted_entities",
        "description": "Report all entities extracted from the source text passages. Call this once with all entities found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "connections": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "details": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["name", "description"],
                    },
                },
                "npcs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "description": {"type": "string"},
                            "personality": {"type": "string"},
                            "knowledge": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "location": {"type": "string"},
                            "relationships": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "source_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["name", "role", "description"],
                    },
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "time_period": {"type": "string"},
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["name", "description"],
                    },
                },
                "factions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "members": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "goals": {"type": "string"},
                            "source_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["name", "description"],
                    },
                },
                "lore": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "culture",
                                    "religion",
                                    "economy",
                                    "geography",
                                    "politics",
                                    "science",
                                    "other",
                                ],
                            },
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "related_entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["category", "title", "content"],
                    },
                },
            },
            "required": ["locations", "npcs", "events", "factions", "lore"],
        },
    }
]


def build_extraction_prompt(chunks: list[Chunk], existing_names: list[str]) -> str:
    """Build the user message for entity extraction."""
    chunk_texts = "\n\n".join(
        f"[Page {c.page or '?'}, Chunk {c.index}]\n{c.text}" for c in chunks
    )
    existing_note = ""
    if existing_names:
        existing_note = (
            f"\n\nAlready extracted entities (avoid duplicates): "
            f"{', '.join(existing_names)}"
        )

    return f"""Analyze the following passages from a book and extract structured world data.

Extract all locations, characters/NPCs, historical events, factions/groups, and cultural/lore details.

For each entity, include the source page numbers where the information was found.
Only extract entities clearly described in the text. Do not invent or speculate.{existing_note}

Use the report_extracted_entities tool to return your findings. If a category has no entities, use an empty array.

--- SOURCE PASSAGES ---
{chunk_texts}
--- END PASSAGES ---"""


def parse_extraction_response(raw: dict) -> dict:
    """Parse and validate the tool_use extraction response."""
    result = {"locations": [], "npcs": [], "events": [], "factions": [], "lore": []}
    for key in result:
        if key in raw and isinstance(raw[key], list):
            result[key] = raw[key]
    return result


def extract_entities_from_chunks(
    chunks: list[Chunk],
    existing_names: list[str] | None = None,
) -> dict:
    """Call Claude with tool_use to extract world entities from text chunks."""
    client = anthropic.Anthropic()
    prompt = build_extraction_prompt(chunks, existing_names or [])

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=EXTRACTION_TOOLS,
        tool_choice={"type": "tool", "name": "report_extracted_entities"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Find the tool_use block in the response
    for block in message.content:
        if block.type == "tool_use" and block.name == "report_extracted_entities":
            return parse_extraction_response(block.input)

    # Fallback: no tool_use block found
    return {"locations": [], "npcs": [], "events": [], "factions": [], "lore": []}
