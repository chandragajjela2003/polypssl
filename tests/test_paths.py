import pytest

from polypssl.paths import detect_environment, get_locations


def test_explicit_environment_overrides_runtime_markers(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setenv("POLYPSSL_ENV", "colab")
    assert detect_environment() == "colab"
    assert get_locations().env == "colab"


def test_invalid_explicit_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("POLYPSSL_ENV", "unknown")
    with pytest.raises(ValueError, match="POLYPSSL_ENV"):
        detect_environment()


def test_location_environment_must_be_supported():
    with pytest.raises(ValueError, match="env must be one of"):
        get_locations("unknown")
