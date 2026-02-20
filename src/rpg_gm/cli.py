import click


@click.group()
def main():
    """RPG Game Master — Turn any book into an explorable world."""
    pass


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", prompt="World name", help="Name for this world")
def ingest(file_path: str, name: str):
    """Ingest a PDF or text file into a playable world."""
    click.echo(f"Ingesting {file_path} as '{name}'...")


@main.command()
@click.argument("world_name")
def play(world_name: str):
    """Play an ingested world."""
    click.echo(f"Loading world '{world_name}'...")
