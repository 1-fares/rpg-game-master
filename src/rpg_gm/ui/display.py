from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rpg_gm.world.models import Location, NPC, World

console = Console()

def show_title(world_title: str):
    console.print()
    console.print(Panel(f"[bold]{world_title}[/bold]", style="bold cyan", box=box.DOUBLE, padding=(1, 4)))
    console.print()

def show_location(location: Location, world: World):
    content = f"[italic]{location.description}[/italic]"
    if location.connections:
        exits = []
        for conn_id in location.connections:
            conn = world.locations.get(conn_id)
            exits.append(conn.name if conn else conn_id)
        content += f"\n\n[dim]Exits:[/dim] {', '.join(exits)}"
    if location.npcs:
        npc_names = []
        for npc_id in location.npcs:
            npc = world.npcs.get(npc_id)
            npc_names.append(npc.name if npc else npc_id)
        content += f"\n[dim]People here:[/dim] {', '.join(npc_names)}"
    if location.details:
        content += f"\n[dim]You notice:[/dim] {', '.join(location.details)}"
    console.print(Panel(content, title=f"[bold green]{location.name}[/bold green]", box=box.ROUNDED))

def show_narration(text: str):
    console.print()
    console.print(Panel(text, style="white", box=box.SIMPLE))

def show_npc_dialogue(npc_name: str, text: str):
    console.print()
    console.print(Panel(text, title=f"[bold yellow]{npc_name}[/bold yellow]", border_style="yellow", box=box.ROUNDED))

def show_journal(entries: list[dict]):
    if not entries:
        console.print("[dim]Your journal is empty.[/dim]")
        return
    table = Table(title="Journal", box=box.SIMPLE_HEAVY)
    table.add_column("Time", style="dim", width=20)
    table.add_column("Entry")
    for entry in entries:
        table.add_row(entry.get("time", ""), entry.get("text", ""))
    console.print(table)

def show_map(visited: set[str], world: World, current_id: str):
    table = Table(title="Known Locations", box=box.ROUNDED)
    table.add_column("Location", style="bold")
    table.add_column("Connections")
    table.add_column("Status")
    for loc_id in sorted(visited):
        loc = world.locations.get(loc_id)
        if not loc:
            continue
        connections = []
        for conn_id in loc.connections:
            conn = world.locations.get(conn_id)
            name = conn.name if conn else conn_id
            if conn_id in visited:
                connections.append(f"[green]{name}[/green]")
            else:
                connections.append(f"[dim]{name}[/dim]")
        marker = "[bold cyan]<< You are here[/bold cyan]" if loc_id == current_id else "[green]Visited[/green]"
        table.add_row(loc.name, ", ".join(connections) if connections else "[dim]none[/dim]", marker)
    console.print(table)

def show_discoveries(discovered: dict, world: World):
    table = Table(title="Discoveries", box=box.ROUNDED)
    table.add_column("Category", style="bold")
    table.add_column("Found")
    table.add_column("Total")
    table.add_column("Progress")
    categories = [
        ("Locations", len(discovered.get("locations", set())), len(world.locations)),
        ("Characters", len(discovered.get("npcs", set())), len(world.npcs)),
        ("Events", len(discovered.get("events", set())), len(world.events)),
        ("Factions", len(discovered.get("factions", set())), len(world.factions)),
        ("Lore", len(discovered.get("lore", set())), len(world.lore)),
    ]
    for name, found, total in categories:
        pct = (found / total * 100) if total > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_len - filled)
        table.add_row(name, str(found), str(total), f"{bar} {pct:.0f}%")
    console.print(table)

def show_entity_for_review(entity_type: str, entity: dict):
    content = ""
    for key, value in entity.items():
        if isinstance(value, list) and value:
            content += f"[bold]{key}:[/bold] {', '.join(str(v) for v in value)}\n"
        elif isinstance(value, dict) and value:
            items = [f"{k}: {v}" for k, v in value.items()]
            content += f"[bold]{key}:[/bold] {'; '.join(items)}\n"
        elif value:
            content += f"[bold]{key}:[/bold] {value}\n"
    title = f"New {entity_type.rstrip('s').title()}"
    console.print(Panel(content.strip(), title=f"[bold magenta]{title}[/bold magenta]", box=box.ROUNDED))
    console.print("[dim][a]ccept  [s]kip  [q]uit extraction[/dim]")

def show_error(msg: str):
    console.print(f"[bold red]Error:[/bold red] {msg}")

def show_info(msg: str):
    console.print(f"[dim]{msg}[/dim]")

def show_help():
    table = Table(title="Commands", box=box.SIMPLE)
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/look", "Describe your current location in detail")
    table.add_row("/map", "Show visited locations and connections")
    table.add_row("/talk <name>", "Start a conversation with someone")
    table.add_row("/examine <thing>", "Examine something in detail")
    table.add_row("/journal", "View your journal")
    table.add_row("/discoveries", "See what you've uncovered")
    table.add_row("/save", "Save your game")
    table.add_row("/load", "Load a saved game")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit the game")
    table.add_row("[dim]anything else[/dim]", "[dim]Describe what you want to do[/dim]")
    console.print(table)

def get_input(prompt_text: str = "> ") -> str:
    try:
        return console.input(f"[bold cyan]{prompt_text}[/bold cyan]").strip()
    except (EOFError, KeyboardInterrupt):
        return "/quit"
