from pathlib import Path

import pytest


@pytest.fixture
def circuit_config():
    return Path(__file__).parent / "data" / "circuit_config.json"


@pytest.fixture
def recipe(tmp_path):
    def _f(extension, recipe_contents):
        fn = tmp_path / f"recipe.{extension}"
        tmp_path.mkdir(parents=True, exist_ok=True)
        with fn.open("w") as fd:
            fd.write(recipe_contents)
        return fn

    return _f
