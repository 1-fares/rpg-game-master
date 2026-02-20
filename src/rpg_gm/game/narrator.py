"""Claude narrator for game narration and NPC dialogue.

Uses tool_use for state changes (move_player, discover_entity, end_conversation)
and includes retry with exponential backoff for API resilience.
"""

import time

import anthropic

from rpg_gm.world.models import NPC, World

# Tools for state changes during narration
NARRATOR_TOOLS = [
    {
        "name": "move_player",
        "description": "Move the player to a new location. Use when the player's action results in traveling to a different place.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location_id": {
                    "type": "string",
                    "description": "The ID (slug) of the destination location",
                },
            },
            "required": ["location_id"],
        },
    },
    {
        "name": "discover_entity",
        "description": "Record that the player has discovered a new piece of world knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["lore", "event", "npc", "faction"],
                },
                "entity_id": {
                    "type": "string",
                    "description": "The ID (slug) of the discovered entity",
                },
            },
            "required": ["entity_type", "entity_id"],
        },
    },
]

NPC_TOOLS = [
    {
        "name": "end_conversation",
        "description": "End the conversation with the player. Use when the player says goodbye or the conversation has reached a natural conclusion.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "discover_entity",
        "description": "Record that the player has learned something new from this conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["lore", "event", "npc", "faction"],
                },
                "entity_id": {"type": "string"},
            },
            "required": ["entity_type", "entity_id"],
        },
    },
]


GM_SYSTEM_TEMPLATE = """You are an immersive game master guiding a player through {world_title}.
You narrate scenes, describe locations, and bring the world to life.

Rules:
- Stay faithful to the source material. Never invent facts not supported by the provided passages.
- When describing a location or event, weave in specific details from the source passages provided in context.
- Write in second person present tense ("You see...", "You hear...").
- Keep responses to 2-4 paragraphs. Rich in sensory detail but not verbose.
- If the player tries to go somewhere that doesn't exist in the world, gently redirect them toward known locations.
- End each narration with subtle hooks — things to examine, people to talk to, paths to explore.

Available locations the player can move to: {available_locations}

Use the move_player tool when the player's action results in traveling to a new location.
Use the discover_entity tool when the player learns something significant.

--- CURRENT WORLD STATE ---
Location: {location_name} — {location_description}
{exits}
{npcs_present}
"""

NPC_SYSTEM_TEMPLATE = """You are {npc_name}, {npc_role}. {npc_description}

Personality: {npc_personality}

You know about: {npc_knowledge}

Rules:
- Stay in character. Speak as this person would — with their vocabulary, concerns, and worldview.
- You can discuss these topics: {npc_knowledge}. For topics outside your knowledge, say you don't know or redirect the conversation.
- Reference specific details from the source passages when answering questions.
- Do not break character. Do not acknowledge being an AI or a game.
- Keep responses to 1-3 paragraphs of natural dialogue.

Use the end_conversation tool when the player says goodbye or wants to leave.
Use the discover_entity tool when you share significant knowledge with the player.
"""


def build_gm_prompt(
    world: World,
    location_id: str,
    passages: list[dict],
    history: list[dict],
    player_input: str,
) -> dict:
    """Build a prompt dict for GM narration."""
    location = world.locations[location_id]

    exits = (
        "Exits: "
        + ", ".join(
            f"{world.locations[c].name} ({c})" if c in world.locations else c
            for c in location.connections
        )
        if location.connections
        else "No obvious exits."
    )

    npc_names = []
    for npc_id in location.npcs:
        npc = world.npcs.get(npc_id)
        if npc:
            npc_names.append(f"{npc.name} ({npc.role})")
    npcs_present = (
        "People here: " + ", ".join(npc_names) if npc_names else "Nobody else is here."
    )

    available_locations = ", ".join(
        f"{loc.name} ({lid})" for lid, loc in world.locations.items()
    )

    system = GM_SYSTEM_TEMPLATE.format(
        world_title=world.title,
        location_name=location.name,
        location_description=location.description,
        exits=exits,
        npcs_present=npcs_present,
        available_locations=available_locations,
    )

    # Build user message with RAG context
    passage_text = ""
    if passages:
        passage_text = "\n\nRELEVANT SOURCE PASSAGES:\n"
        for p in passages:
            page_ref = f" (p.{p['page']})" if p.get("page") else ""
            passage_text += f"- {p['text']}{page_ref}\n"

    user_content = f"{passage_text}\n\nPLAYER ACTION: {player_input}"

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    return {"system": system, "messages": messages, "tools": NARRATOR_TOOLS}


def build_npc_prompt(
    npc: NPC,
    passages: list[dict],
    history: list[dict],
    player_input: str,
) -> dict:
    """Build a prompt dict for NPC dialogue."""
    system = NPC_SYSTEM_TEMPLATE.format(
        npc_name=npc.name,
        npc_role=npc.role,
        npc_description=npc.description,
        npc_personality=npc.personality,
        npc_knowledge=", ".join(npc.knowledge),
    )

    passage_text = ""
    if passages:
        passage_text = "\n\nRELEVANT SOURCE PASSAGES:\n"
        for p in passages:
            page_ref = f" (p.{p['page']})" if p.get("page") else ""
            passage_text += f"- {p['text']}{page_ref}\n"

    user_content = f"{passage_text}\n\nThe player says: {player_input}"

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    return {"system": system, "messages": messages, "tools": NPC_TOOLS}


def narrate(prompt: dict, max_retries: int = 3, client: anthropic.Anthropic | None = None) -> anthropic.types.Message:
    """Send prompt to Claude and return the full Message object.

    Contains text + tool_use blocks. Retries on rate limit and API errors
    with exponential backoff.
    """
    client = client or anthropic.Anthropic()
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=prompt["system"],
                messages=prompt["messages"],
                tools=prompt.get("tools", []),
            )
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 2**attempt
                time.sleep(delay)
            else:
                raise
    raise last_error  # Should not reach here


def narrate_stream(prompt: dict, max_retries: int = 3, client: anthropic.Anthropic | None = None):
    """Send prompt to Claude and yield text chunks, then tool_use blocks.

    Yields tuples of (type, data):
    - ("text", str) for text chunks
    - ("tool_use", dict) for tool calls (name + input), yielded at the end

    Retries on rate limit and API errors with exponential backoff.
    """
    client = client or anthropic.Anthropic()
    last_error = None
    for attempt in range(max_retries):
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=prompt["system"],
                messages=prompt["messages"],
                tools=prompt.get("tools", []),
            ) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield ("text", event.delta.text)

                # After streaming, get the final message for tool_use blocks
                final_message = stream.get_final_message()
                for block in final_message.content:
                    if block.type == "tool_use":
                        yield ("tool_use", {"name": block.name, "input": block.input})
                return  # Success, don't retry
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 2**attempt
                time.sleep(delay)
            else:
                raise
    raise last_error


def extract_text_and_tools(
    message: anthropic.types.Message,
) -> tuple[str, list[dict]]:
    """Extract text content and tool_use calls from a Message."""
    text_parts = []
    tool_calls = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"name": block.name, "input": block.input})
    return "\n".join(text_parts), tool_calls
