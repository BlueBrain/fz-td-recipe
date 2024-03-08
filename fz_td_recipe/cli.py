"""The command line interface."""

from pathlib import Path

import click

from .legacy import XMLRecipe
from .recipe import Recipe
from .transform import _RecipeEncoder


@click.group()
def app():
    """To be used to provide commands."""


@app.command()
@click.option("--circuit-config", type=click.Path(exists=True))
@click.option("--from", "src")
@click.option("--to", "dst")
@click.argument("filename", type=click.Path(exists=True))
def validate(filename, circuit_config=None, src=None, dst=None):
    """Simple recipe validation."""
    Recipe(Path(filename), circuit_config, (src, dst))


@app.command()
@click.option("--circuit-config", type=click.Path(exists=True))
@click.option("--from", "src")
@click.option("--to", "dst")
@click.argument("xml_file", type=click.Path(exists=True))
@click.argument("json_file", type=click.Path())
def convert(xml_file, json_file, circuit_config=None, src=None, dst=None):
    """Convert an XML recipe into a JSON equivalent."""
    nodes = None
    if circuit_config:
        nodes = (src, dst)
    recipe = XMLRecipe(xml_file, strict=False)
    with open(json_file, "w", encoding="utf8") as fd:
        fd.write(_RecipeEncoder(Path(json_file).parent, circuit_config, nodes).encode(recipe))
