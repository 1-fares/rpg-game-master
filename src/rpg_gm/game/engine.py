from pathlib import Path

import anthropic

from rpg_gm.world.models import World
from rpg_gm.world.loader import load_world, get_world_dir
from rpg_gm.game.state import GameState
from rpg_gm.game.commands import parse_input, find_npc_by_name, find_location_by_name
from rpg_gm.game.narrator import (
    build_gm_prompt,
    build_npc_prompt,
    narrate_stream,
    NARRATOR_TOOLS,
    NPC_TOOLS,
)
from rpg_gm.ingestion.embedder import query_chunks
from rpg_gm.ui.display import (
    show_title,
    show_location,
    show_narration,
    show_npc_dialogue,
    show_journal,
    show_map,
    show_discoveries,
    show_help,
    show_error,
    show_info,
    get_input,
    console,
)

MAX_CONVERSATION_TURNS = 20


class GameEngine:
    def __init__(self, world_name: str, base_dir: Path | None = None):
        self.world_name = world_name
        self.base_dir = base_dir or Path("worlds")
        self.world = load_world(world_name, base_dir=self.base_dir)
        self.world_dir = get_world_dir(world_name, base_dir=self.base_dir)

        # Set up game state
        saves_dir = self.world_dir / "saves"
        saves_dir.mkdir(exist_ok=True)
        self.state = GameState(str(saves_dir / "autosave.db"))

        # Pick a starting location (first location in the world)
        if not self.world.locations:
            raise ValueError(f"World '{world_name}' has no locations. Re-run ingestion.")
        start = next(iter(self.world.locations))
        self.state.init(world_name, start)

        # Reusable API client
        self.client = anthropic.Anthropic()

        # Conversation tracking
        self.talking_to: str | None = None  # NPC ID if in conversation
        self.conversation_turns: int = 0

    def run(self):
        """Main game loop."""
        show_title(self.world.title)
        self._discover_current_location()
        self._show_current_location()
        show_info("Type /help for commands, or describe what you want to do.")
        console.print()

        try:
            while True:
                prompt_text = f"[{self.world.npcs[self.talking_to].name}] > " if self.talking_to else "> "
                raw_input = get_input(prompt_text)

                if not raw_input:
                    continue

                cmd = parse_input(raw_input)

                if cmd:
                    # All slash commands work regardless of conversation state
                    should_quit = self._handle_command(cmd)
                    if should_quit:
                        break
                else:
                    self._handle_free_input(raw_input)

            show_info("Game saved. Farewell, traveler.")
        finally:
            self.state.close()

    def _handle_command(self, cmd) -> bool:
        """Handle a slash command. Returns True if the game should quit."""
        if cmd.name == "quit":
            return True
        elif cmd.name == "help":
            show_help()
        elif cmd.name == "look":
            self._do_look()
        elif cmd.name == "map":
            show_map(self.state.get_visited(), self.world, self.state.get_location())
        elif cmd.name == "talk":
            self._do_talk(cmd.args)
        elif cmd.name == "journal":
            show_journal(self.state.get_journal())
        elif cmd.name == "discoveries":
            show_discoveries(self.state.get_discovered(), self.world)
        elif cmd.name == "examine":
            self._do_examine(cmd.args)
        elif cmd.name == "save":
            show_info("Game auto-saves continuously. Your progress is safe.")
        elif cmd.name == "load":
            show_info("Game loaded from last autosave on startup.")
        elif cmd.name in ("end", "bye", "leave"):
            if self.talking_to:
                npc = self.world.npcs[self.talking_to]
                show_info(f"You end your conversation with {npc.name}.")
                self.talking_to = None
                self.conversation_turns = 0
            else:
                show_info("You're not in a conversation.")
        else:
            show_error(f"Unknown command: /{cmd.name}. Type /help for available commands.")
        return False

    def _handle_free_input(self, player_input: str):
        """Handle free-text input — either conversation or exploration."""
        if self.talking_to:
            self._do_npc_conversation(player_input)
        else:
            self._do_explore(player_input)

    def _do_look(self):
        """Detailed look at current location using RAG."""
        location_id = self.state.get_location()
        location = self.world.locations[location_id]
        passages = self._query_rag(f"description of {location.name}")
        prompt = build_gm_prompt(
            world=self.world,
            location_id=location_id,
            passages=passages,
            history=[],
            player_input=f"Describe {location.name} in vivid detail. What do I see, hear, and smell?",
        )
        self._stream_and_handle(prompt, is_npc=False)

    def _do_talk(self, npc_name: str):
        """Start a conversation with an NPC."""
        if not npc_name:
            location = self.world.locations[self.state.get_location()]
            if location.npcs:
                npc_names = [self.world.npcs[nid].name for nid in location.npcs if nid in self.world.npcs]
                show_info(f"People here: {', '.join(npc_names)}")
            else:
                show_info("There's nobody here to talk to.")
            return

        location = self.world.locations[self.state.get_location()]
        npc_id = find_npc_by_name(npc_name, location.npcs, self.world)

        if not npc_id:
            all_npc_id = find_npc_by_name(npc_name, list(self.world.npcs.keys()), self.world)
            if all_npc_id:
                npc = self.world.npcs[all_npc_id]
                loc = self.world.locations.get(npc.location_id)
                loc_name = loc.name if loc else npc.location_id
                show_error(f"{npc.name} is not here. They can be found at {loc_name}.")
            else:
                show_error(f"Nobody named '{npc_name}' is known to you.")
            return

        self.talking_to = npc_id
        self.conversation_turns = 0
        npc = self.world.npcs[npc_id]
        self.state.discover("npcs", npc_id)
        self.state.add_journal_entry(f"Began speaking with {npc.name}.")
        show_info(f"You approach {npc.name}. (Type /end to leave the conversation)")

        # Opening greeting from NPC
        passages = self._query_rag(f"{npc.name} {npc.role}")
        prompt = build_npc_prompt(
            npc=npc,
            passages=passages,
            history=[],
            player_input="I approach you and greet you.",
        )
        self._stream_and_handle(prompt, is_npc=True, npc_name=npc.name)

    def _do_examine(self, thing: str):
        if not thing:
            show_info("Examine what? Try /examine <something>")
            return
        location = self.world.locations[self.state.get_location()]
        passages = self._query_rag(f"{thing} at {location.name}")
        prompt = build_gm_prompt(
            world=self.world,
            location_id=self.state.get_location(),
            passages=passages,
            history=[],
            player_input=f"I examine {thing} closely. Describe it in detail.",
        )
        self._stream_and_handle(prompt, is_npc=False)

    def _do_explore(self, player_input: str):
        """Handle free-text exploration input."""
        location_id = self.state.get_location()
        passages = self._query_rag(player_input)
        history = self.state.get_recent_messages(limit=10)
        prompt = build_gm_prompt(
            world=self.world,
            location_id=location_id,
            passages=passages,
            history=history,
            player_input=player_input,
        )
        self.state.add_message("user", player_input)
        self._stream_and_handle(prompt, is_npc=False)

    def _do_npc_conversation(self, player_input: str):
        """Handle free-text input during NPC conversation."""
        self.conversation_turns += 1
        if self.conversation_turns >= MAX_CONVERSATION_TURNS:
            npc = self.world.npcs[self.talking_to]
            show_info(f"{npc.name} seems to have other matters to attend to. The conversation ends.")
            self.talking_to = None
            self.conversation_turns = 0
            return

        npc = self.world.npcs[self.talking_to]
        passages = self._query_rag(f"{npc.name} {player_input}")
        history = self.state.get_recent_messages(limit=10)
        prompt = build_npc_prompt(
            npc=npc,
            passages=passages,
            history=history,
            player_input=player_input,
        )
        self.state.add_message("user", player_input)
        self._stream_and_handle(prompt, is_npc=True, npc_name=npc.name)

    def _stream_and_handle(self, prompt: dict, is_npc: bool = False, npc_name: str = ""):
        """Stream Claude's response, display it, and handle tool calls."""
        full_text = ""
        tool_calls = []

        console.print()
        if is_npc and npc_name:
            console.print(f"[bold yellow]{npc_name}:[/bold yellow]")

        try:
            for event_type, data in narrate_stream(prompt, client=self.client):
                if event_type == "text":
                    console.print(data, end="")
                    full_text += data
                elif event_type == "tool_use":
                    tool_calls.append(data)
        except Exception as e:
            show_error(f"{'Dialogue' if is_npc else 'Narration'} failed: {e}")
            return

        console.print()
        console.print()

        # Store assistant response
        if full_text:
            self.state.add_message("assistant", full_text)

        # Process tool calls
        for tc in tool_calls:
            self._handle_tool_call(tc)

    def _handle_tool_call(self, tool_call: dict):
        """Process a tool call from Claude's response."""
        name = tool_call["name"]
        inp = tool_call.get("input", {})

        if name == "move_player":
            location_id = inp.get("location_id", "")
            if location_id in self.world.locations:
                self.state.set_location(location_id)
                self._discover_current_location()
                self._show_current_location()
                loc_name = self.world.locations[location_id].name
                self.state.add_journal_entry(f"Traveled to {loc_name}.")
            # Silently ignore invalid location IDs

        elif name == "discover_entity":
            entity_type = inp.get("entity_type", "lore")
            entity_id = inp.get("entity_id", "")
            # Normalize singular→plural to match get_discovered bucket names
            pluralize = {"npc": "npcs", "faction": "factions", "event": "events"}
            entity_type = pluralize.get(entity_type, entity_type)
            if entity_id:
                self.state.discover(entity_type, entity_id)

        elif name == "end_conversation":
            if self.talking_to:
                npc = self.world.npcs[self.talking_to]
                show_info(f"{npc.name} nods farewell.")
                self.talking_to = None
                self.conversation_turns = 0

    def _query_rag(self, query: str) -> list[dict]:
        try:
            return query_chunks(
                query=query,
                world_name=self.world_name,
                persist_dir=self.world_dir / "chroma",
                n_results=5,
            )
        except Exception:
            return []

    def _discover_current_location(self):
        loc_id = self.state.get_location()
        self.state.discover("locations", loc_id)

    def _show_current_location(self):
        loc_id = self.state.get_location()
        location = self.world.locations[loc_id]
        show_location(location, self.world)
