import pytest

from polypssl.config import load_config


def test_legacy_false_switches_mean_disabled_modes(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "model: {name: polyp_pvt, ual: false}\n"
        "fal: {mode: false}\n"
        "data: {augment: false}\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg["model"]["ual"] == "off"
    assert cfg["fal"]["mode"] == "off"
    assert cfg["data"]["augment"] == "none"


def test_true_boolean_ual_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("model: {name: polyp_pvt, ual: true}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model.ual"):
        load_config(path)
