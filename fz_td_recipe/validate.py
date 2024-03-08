"""Provide basic recipe validation."""

import json
from pathlib import Path

import jsonschema
import yaml

SCHEMA = Path(__file__).parent / "data" / "schema.yaml"


def validate_recipe(path: Path):
    """Load the recipe given as `path` and validate it."""
    with path.open() as fd:
        recipe = json.load(fd)
    with SCHEMA.open() as fd:
        schema = yaml.safe_load(fd)
    jsonschema.validate(instance=recipe, schema=schema)
