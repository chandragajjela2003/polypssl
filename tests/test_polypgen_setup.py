"""PolypGen extraction (setup_env.extract_polypgen) and loading (discover.polypgen) on a fake zip."""
import zipfile

import cv2
import numpy as np
import pytest

import setup_env
from polypssl.data import test_sets as all_test_sets
from polypssl.data.discover import polypgen


def _png(arr):
    return cv2.imencode(".jpg", arr)[1].tobytes()


def _fake_zip(path, centres=("C1", "C2", "C3", "C5", "C6"), n=3, top="PolypGen2021_MultiCenterData_v3"):
    img = np.full((40, 50, 3), 100, np.uint8)
    mask = np.zeros((40, 50), np.uint8)
    mask[10:20, 10:20] = 255
    with zipfile.ZipFile(path, "w") as z:
        for c in centres:
            for i in range(n):
                z.writestr(f"{top}/data_{c}/images_{c}/{c}_{i:03d}.jpg", _png(img))
                z.writestr(f"{top}/data_{c}/masks_{c}/{c}_{i:03d}_mask.jpg", _png(mask))
            z.writestr(f"{top}/data_{c}/bbox_{c}/{c}_000.txt", "0 1 2 3")
        # video sequences must be ignored
        z.writestr(f"{top}/sequenceData/positive/seq1/images_seq1/f1.jpg", _png(img))
        z.writestr(f"{top}/sequenceData/positive/seq1/masks_seq1/f1_mask.jpg", _png(mask))
    return path


def test_extract_and_load_polypgen(tmp_path):
    counts = setup_env.extract_polypgen(_fake_zip(tmp_path / "pg.zip"), tmp_path / "data")
    assert counts == {"C1": 3, "C2": 3, "C3": 3, "C5": 3, "C6": 3}
    samples = polypgen(tmp_path / "data", "C1")
    assert len(samples) == 3 and samples[0]["mask"].endswith("_mask.jpg")
    assert not (tmp_path / "data" / "PolypGen" / "sequenceData").exists()


def test_ambiguous_or_unpaired_centres_are_skipped(tmp_path):
    z = _fake_zip(tmp_path / "pg.zip", centres=("C1",))
    with zipfile.ZipFile(z, "a") as zz:                       # C1 also appears elsewhere -> ambiguous
        zz.writestr("copy/images_C1/x.jpg", b"x")
        zz.writestr("PolypGen2021_MultiCenterData_v3/data_C2/images_C2/a.jpg", b"x")   # C2: no masks
    counts = setup_env.extract_polypgen(z, tmp_path / "data")
    assert counts == {}


def test_prepare_polypgen_never_stops_setup(tmp_path, monkeypatch):
    from polypssl.paths import Locations
    loc = Locations("kaggle", tmp_path / "data", tmp_path / "out")
    monkeypatch.setattr(setup_env, "zip_store", lambda env, loc: tmp_path / "store")
    setup_env.prepare_polypgen("kaggle", loc)                 # no zip: only a message
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / setup_env.POLYPGEN_ZIP).write_bytes(b"not a zip")
    setup_env.prepare_polypgen("kaggle", loc)                 # broken zip: message, no exception
    assert not (tmp_path / "data" / "PolypGen").exists()


def test_extracted_polypgen_appears_in_test_sets(fake_data, tmp_path):
    setup_env.extract_polypgen(_fake_zip(tmp_path / "pg.zip"), fake_data)
    sets = all_test_sets(fake_data, include_external=True, expect_pranet_counts=False)
    assert {f"PolypGen-{c}" for c in ("C1", "C2", "C3", "C5", "C6")} <= set(sets)
