"""Test synapse property mapping
"""

from io import StringIO
from fz_td_recipe import XMLRecipe


def test_synapse_properties():
    r = XMLRecipe(StringIO(RECIPE))
    assert set(r.synapse_properties.rules.required) == set(
        ["fromRegion", "fromHemisphere", "toRegion", "toHemisphere"]
    )


RECIPE = """\
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
"""
