"""Test synapse property mapping"""

import json
from io import StringIO

import pytest
from jsonschema.exceptions import ValidationError

from fz_td_recipe import Recipe, XMLRecipe


def test_synapse_properties():
    r = XMLRecipe(StringIO(RECIPE_XML))
    assert set(r.synapse_properties.rules.required) == set(
        ["fromRegion", "fromHemisphere", "toRegion", "toHemisphere"]
    )


def test_valid_synapse_properties(circuit_config, tmp_path):
    data = json.loads(RECIPE_JSON)

    recipe_file = tmp_path / "recipe.json"
    with recipe_file.open("w") as fd:
        json.dump(data, fd)
    Recipe(recipe_file, circuit_config, (None, None))


@pytest.mark.parametrize(
    "parameter",
    [
        "decay_time_sd",
        "u_syn_sd",
        "depression_time_mu",
        "depression_time_sd",
        "facilitation_time_mu",
        "facilitation_time_sd",
        "conductance_mu",
        "conductance_sd",
    ],
)
def test_invalid_synapse_properties(circuit_config, tmp_path, parameter):
    data = json.loads(RECIPE_JSON)
    data["synapse_properties"]["classes"][0][parameter] = 0.0

    recipe_file = tmp_path / "recipe.json"
    with recipe_file.open("w") as fd:
        json.dump(data, fd)
    with pytest.raises(ValidationError, match=parameter):
        Recipe(recipe_file, circuit_config, (None, None))


RECIPE_JSON = """\
{
  "bouton_distances": {},
  "gap_junction_properties": {},
  "seed": 0,
  "synapse_properties": {
    "rules": [
      {
        "src_mtype": "*",
        "class": "I1"
      }
    ],
    "classes": [
      {
        "class": "I1",
        "n_rrp_vesicles_mu": 1,
        "conductance_mu": 1.0,
        "conductance_sd": 0.1,
        "decay_time_mu": 8.3,
        "decay_time_sd": 2.2,
        "u_syn_mu": 0.25,
        "u_syn_sd": 0.13,
        "depression_time_mu": 706.0,
        "depression_time_sd": 405.0,
        "facilitation_time_mu": 21.0,
        "facilitation_time_sd": 9.0
      }
    ]
  },
  "version": 1
}
"""


RECIPE_XML = """\
<?xml version="1.0"?>
<blueColumn>
  <SynapsesProperties>
    <synapse type="I1" />
    <synapse fromRegion="Foo" fromHemisphere="left" type="I2" />
    <synapse toRegion="Bar" toHemisphere="right" type="I3" />
  </SynapsesProperties>
  <SynapsesClassification>
    <class id="I1"  gsyn="0.0" gsynSD="0.0" dtc="8.30" dtcSD="2.2" u="0.25" uSD="0.13" d="706" dSD="405" f="021" fSD="9" />
    <class id="I2"  gsyn="1.0" gsynSD="0.0" dtc="1.74" dtcSD="0.18" u="0.0" uSD="0.50" d="671" dSD="017" f="017" fSD="5" />
    <class id="I3"  gsyn="2.0" gsynSD="0.0" dtc="8.30" dtcSD="2.2" u="0.25" uSD="0.13" d="706" dSD="405" f="021" fSD="9" />
  </SynapsesClassification>
</blueColumn>
"""  # noqa: E501
