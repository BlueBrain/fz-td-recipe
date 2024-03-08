from fz_td_recipe.transform import _snake_case


def test_snake_case():
    assert _snake_case("MType") == "mtype"
    assert _snake_case("mType") == "m_type"
    assert _snake_case("fromMType") == "from_mtype"
    assert _snake_case("toMType") == "to_mtype"
