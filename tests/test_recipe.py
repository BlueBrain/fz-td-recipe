"""Test recipe class.
"""

from fz_td_recipe import Recipe

import pytest


def test_minimal(recipe, circuit_config):
    r = Recipe(recipe("json", "{}"), circuit_config, (None, None))

    assert r.get("bouton_distances.excitatory_synapse_distance") == 25.0
    assert r.get("gap_junction_properties.conductance") == 0.2
    assert r.get("seed") == 0

    r.set("gap_junction_properties.conductance", 1.23)
    assert r.get("gap_junction_properties.conductance") == 1.23


def test_simple(recipe, circuit_config):
    r = Recipe(recipe("yaml", RECIPE_SIMPLE), circuit_config, (None, None))

    assert r.get("bouton_distances.excitatory_synapse_distance") == 55.0
    assert r.get("bouton_distances.inhibitory_synapse_distance") == 5.0
    assert r.get("seed") == 123

    assert r.get_values(["dst_mtype"]) == {"dst_mtype": ["macron", "macaron", "macaroon"]}

    columns, matrix = r.as_matrix("touch_rules")

    assert matrix[0] == 0
    assert matrix[1] == 1
    assert matrix[2] == 1


def test_simple_not_allowed(recipe, circuit_config):
    r = Recipe(recipe("yaml", RECIPE_SIMPLE), circuit_config, (None, None))

    r.set("bar", 1)
    with pytest.raises(ValueError):
        r.set("spam.ham", 2)

    with pytest.raises(KeyError):
        r.get_values(["afferent_foo"])


def test_simple_invalid_mtype(recipe, circuit_config):
    r = Recipe(
        recipe("yaml", RECIPE_SIMPLE + RECIPE_SIMPLE_INVALID_ADDENDUM), circuit_config, (None, None)
    )

    with pytest.raises(ValueError, match=".*untranslated.*"):
        r.as_pandas("touch_rules")


def test_simple_load_invalid(recipe, circuit_config):
    with pytest.raises(ValueError, match=".*needs to be JSON.*"):
        Recipe(recipe("yahml", RECIPE_SIMPLE), circuit_config, (None, None))


RECIPE_SIMPLE = """
seed: 123
bouton_distances:
    excitatory_synapse_distance: 55.0
touch_rules:
    - {src_mtype: "*", dst_mtype: "*", afferent_section_type: "*", efferent_section_type: "soma"}
"""

RECIPE_SIMPLE_INVALID_ADDENDUM = """
    - {src_mtype: "mitterand", dst_mtype: "*", afferent_section_type: "*", efferent_section_type: "soma"}
"""
