from click.testing import CliRunner
from pandas.testing import assert_frame_equal

from fz_td_recipe.cli import app
from fz_td_recipe.recipe import Recipe


def test_convert(circuit_config, recipe, tmp_path):
    xml_path = recipe("xml", RECIPE_BASIC_XML)
    json_path = tmp_path / "converted.json"
    runner = CliRunner()
    result = runner.invoke(app, ["convert", str(xml_path), str(json_path)])

    assert result.exit_code == 0
    assert result.output == ""

    r = Recipe(json_path, circuit_config, (None, None))

    assert r.get("seed") == 0

    df = r.as_pandas("touch_rules")

    assert list(df.columns) == ["src_mtype_i", "dst_mtype_i", "afferent_section_type"]

    cols, matrix = r.as_matrix("touch_rules")

    assert cols == ["afferent_section_type", "dst_mtype_i"]

    full_json_path = tmp_path / "with_parquet" / "converted.json"
    full_json_path.parent.mkdir(exist_ok=True, parents=True)
    result = runner.invoke(
        app,
        ["convert", "--circuit-config", str(circuit_config), str(xml_path), str(full_json_path)],
    )

    assert result.exit_code == 0
    assert result.output == ""

    r2 = Recipe(full_json_path, circuit_config, (None, None))

    parquet_path = tmp_path / "with_parquet" / "synapse_rules.parquet"
    assert parquet_path.exists()

    cols1, matrix1 = r.as_matrix("synapse_properties.rules")
    cols2, matrix2 = r2.as_matrix("synapse_properties.rules")

    assert cols1 == cols2

    df1 = r.as_pandas("synapse_properties.rules")
    df2 = r2.as_pandas("synapse_properties.rules")

    assert set(df1.columns) == set(df2.columns)
    df2 = df2[df1.columns]
    assert_frame_equal(df1, df2)


RECIPE_BASIC_XML = """
<recipe>
  <TouchRules>
    <touchRule fromMType="*" toMType="*" toBranchType="dendrite" />
    <touchRule fromMType="*" toMType="m*ron" toBranchType="soma" />
  </TouchRules>
  <SynapsesProperties axonalConductionVelocity="313">
    <synapse fromRegion="*am" type="EEE" />
  </SynapsesProperties>
  <SynapsesClassification>
    <class id="EEE" gsyn="1" gsynSD="2" nrrp="3" dtc="4" dtcSD="5" u="6" uSD="7" d="8" dSD="9" f="10" fSD="11" />
  </SynapsesClassification>
</recipe>
"""
