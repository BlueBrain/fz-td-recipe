"""Test the seed part"""

from io import StringIO

from fz_td_recipe import XMLRecipe


def test_default_properties():
    r = XMLRecipe(StringIO(PROPERTIES_ABSENT))
    assert r.gap_junction_properties.gsyn == 0.2
    assert str(r) == PROPERTIES_ABSENT


def test_full_properties():
    r = XMLRecipe(StringIO(PROPERTIES_FULL))
    assert r.gap_junction_properties.gsyn == 0.9
    assert str(r) == PROPERTIES_FULL

    assert r.bouton_distances.validate()


PROPERTIES_ABSENT = """\
<?xml version="1.0"?>
<blueColumn>
</blueColumn>
"""

PROPERTIES_FULL = """\
<?xml version="1.0"?>
<blueColumn>
  <GapJunctionProperty gsyn="0.9" />
</blueColumn>
"""
