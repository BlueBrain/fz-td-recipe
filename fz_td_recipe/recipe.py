"""Top-level interface for recipe handling."""

import json
from pathlib import Path

import jsonschema
import pandas as pd
import yaml

from .transform import _Pandifier

SCHEMA = Path(__file__).parent / "data" / "schema.yaml"
DEFAULTS = Path(__file__).parent / "data" / "defaults.yaml"


PARQUET_CANDIDATES = ["synapse_properties.rules"]


def _apply_defaults(settings, defaults):
    if isinstance(defaults, (list, tuple)) and len(defaults) > 0:
        (value,) = defaults
        if isinstance(settings, (list, tuple)):
            for item in settings:
                _apply_defaults(item, value)
        elif isinstance(settings, pd.DataFrame):
            for k, v in value.items():
                if k not in settings.columns:
                    settings[k] = v
    elif isinstance(defaults, dict):
        for k, v in defaults.items():
            if k not in settings:
                settings[k] = v
            elif isinstance(v, (dict, list, tuple)):
                _apply_defaults(settings[k], v)


class Recipe:
    """Wrapper to load and validate a JSON or YAML recipe."""

    def __init__(self, filename: Path, circuit_config, nodes):
        """Open a recipe in `filename` using the corresponding circuit config.

        The `filename` argument may point either to a JSON file ending in `.json` or a
        YAML file ending in either `.yml` or `.yaml`.
        """
        with filename.open() as fd:
            if filename.suffix == ".json":
                self.__recipe = json.load(fd)
            elif filename.suffix in (".yml", ".yaml"):
                self.__recipe = yaml.safe_load(fd)
            else:
                raise ValueError("Recipe needs to be JSON or YAML")

        with DEFAULTS.open() as fd:
            defaults = yaml.safe_load(fd)

        with SCHEMA.open() as fd:
            schema = yaml.safe_load(fd)

        jsonschema.validate(instance=defaults, schema=schema)
        jsonschema.validate(instance=self.__recipe, schema=schema)

        for key in PARQUET_CANDIDATES:
            if (value := self.get(key)) and isinstance(value, str):
                self.set(key, pd.read_parquet(filename.parent / value))

        _apply_defaults(self.__recipe, defaults)

        self._pandifier = _Pandifier(circuit_config, nodes)

    def get(self, key):
        """Get a recipe component under `key`.

        Nested values are accessed by joining all keys with a period.
        """
        value = self.__recipe
        for component in key.split("."):
            if component in value:
                value = value[component]
            else:
                return None
        return value

    def get_values(self, columns):
        """Return a dictionary with all values of the specified columns.

        Dictionary keys are the column names, values are taken from the corresponding node
        population of the circuit configuration passed to the constructor.
        """
        return self._pandifier.get_values(columns)

    def set(self, key, value):
        """Set a recipe component under `key`.

        Nested values are set by joining all keys with a period.
        """
        container = self.__recipe
        *parts, last = key.split(".")
        for part in parts:
            if part in container:
                container = container[part]
            else:
                raise ValueError(f"{key} cannot be set")
        container[last] = value

    def as_matrix(self, component):
        """Construct a rule assignment matrix.

        Takes a list of rules, determines which columns relating to the source and target
        population are of essence (contain more than just general wildcards `*`).  The
        enumeration value count in the corresponding node population `@libray` enumeration
        then is used to construct a matrix, where the entries in the matrix set depending
        on the numerical values of the columns, with wildcards expanded.

        Returns a tuple containing the list of columns used (ordered corresponding to the
        matrix dimensions) and the matrix itself.
        """
        return self._pandifier.matrify(self.get(component))

    def as_pandas(self, component):
        """Converts the rules in `component` to a Pandas DataFrame.

        All partial wildcards will be expanded, and columns relating to source or target
        population will be converted to numerical values corresponding to the `@library`
        enumeration entries in the node populations.
        """
        comp = self.get(component)
        if isinstance(comp, pd.DataFrame):
            return comp
        return self._pandifier.framify(comp)
