"""Mean Teacher, ZoomMatch views and Stage D label selection."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from polypssl.config import load_config
from polypssl.data import pranet_train
from polypssl.paths import Locations
from polypssl.ssl_views import _grid_sizes, strong_photometric, zoom_mosaic


# ---------------------------------------------------------------- views
def _square_batch(b=4, h=64, side=(30, 20, 12, 40)):
    x = torch.zeros(b, 3, h, h)
    y = torch.zeros(b, 1, h, h)
    for i in range(b):
        s = side[i % len(side)]
        x[i, :, 5:5 + s, 7:7 + s] = 0.9
        y[i, :, 5:5 + s, 7:7 + s] = 1.0
    return x, y


def test_zoom_mosaic_keeps_image_and_label_aligned():
    x, y = _square_batch()
    cfg = {"prob": 1.0, "area_min": 0.003, "area_max": 0.15, "max_grid": 4}
    for mode in ("targeted", "fixed2"):
        for seed in range(5):
            zx, zy, zw, grids = zoom_mosaic(x, y, None, mode, cfg, torch.Generator().manual_seed(seed))
            assert zx.shape == x.shape and zy.shape == y.shape and zw.shape == y.shape
            # image = 0.9 x mask, so after identical placement and resizing the image must still be 0.9 x label
            assert torch.allclose(zx[:, :1], 0.9 * zy, atol=1e-5), (mode, seed, grids)
            assert torch.all(zw == 1)                                      # no weights given -> all ones
            if mode == "fixed2":
                assert grids == [2, 2, 2, 2]


def test_targeted_zoom_shrinks_polyp_area_towards_target():
    x, y = _square_batch(side=(40,))                                       # every polyp 39 % of the image
    cfg = {"prob": 1.0, "area_min": 0.003, "area_max": 0.15, "max_grid": 4}
    _, zy, _, grids = zoom_mosaic(x, y, None, "targeted", cfg, torch.Generator().manual_seed(0))
    assert all(g > 1 for g in grids)
    a0 = (y >= 0.5).float().mean().item()
    for i, g in enumerate(grids):                                          # own tile shrinks by g^2
        assert math.isclose(a0 / g ** 2, (40 // g) ** 2 / 64 ** 2, rel_tol=0.35)


def test_grid_rule():
    cfg = {"prob": 1.0, "area_min": 0.01, "area_max": 0.01, "max_grid": 4}     # target f fixed at 1 %
    g = torch.Generator().manual_seed(0)
    assert _grid_sizes(torch.tensor([0.04, 0.09, 0.005, 0.9]), "targeted", cfg, g) == [2, 3, 1, 4]
    assert _grid_sizes(torch.tensor([0.5]), "targeted", {**cfg, "prob": 0.0}, g) == [1]


def test_zoom_weights_follow_placement_and_border_is_certain_background():
    x, y = _square_batch(b=2)
    w = torch.zeros_like(y)                                                # all pixels "uncertain"
    cfg = {"prob": 1.0, "area_min": 0.003, "area_max": 0.15, "max_grid": 3}
    x = x[:, :, :63, :63]; y = y[:, :, :63, :63]; w = w[:, :, :63, :63]   # 63 = 3*21: no border at g=3
    for seed in range(5):
        _, zy, zw, grids = zoom_mosaic(x, y, w, "fixed2", cfg, torch.Generator().manual_seed(seed))
        assert torch.all(zw[:, :, :62, :62] == 0)                          # tiles keep the zero weights
        assert torch.all(zw[:, :, 62:, :] == 1) and torch.all(zy[:, :, 62:, :] == 0)   # 1-px border


def test_strong_photometric_is_pixel_aligned_and_bounded():
    x, _ = _square_batch()
    out = strong_photometric(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, x)


# ---------------------------------------------------------------- Mean Teacher pieces
def test_ema_update_and_rampup():
    from polypssl.engine.semi import _ema_update, _rampup
    s, t = torch.nn.BatchNorm1d(3), torch.nn.BatchNorm1d(3)
    with torch.no_grad():
        s.weight.fill_(1.0); t.weight.fill_(0.0); s.running_mean.fill_(5.0)
    _ema_update(t, s, step=0, decay=0.99)                                  # step 0: alpha = 0 -> copy
    assert torch.allclose(t.weight, s.weight) and torch.allclose(t.running_mean, s.running_mean)
    with torch.no_grad():
        t.weight.fill_(0.0)
    _ema_update(t, s, step=1000, decay=0.99)
    assert torch.allclose(t.weight, torch.full((3,), 0.01))
    assert _rampup(0, 1000, 0.1) == pytest.approx(math.exp(-5))
    assert _rampup(100, 1000, 0.1) == 1.0 and _rampup(900, 1000, 0.1) == 1.0
    assert _rampup(30, 1000, 0.1) < _rampup(60, 1000, 0.1)


def _semi_setup(fake_data, tmp_path, monkeypatch, zoom="targeted", iterations=4, name="mt"):
    """Split with 3 labelled train, 1 val, 6 unlabelled; the unlabelled MASK FILES ARE DELETED, so the run
    fails if Mean Teacher ever reads an unlabelled mask."""
    import polypssl.engine.semi as semi
    samples = pranet_train(fake_data, expect_counts=False)
    ids = sorted(s["id"] for s in samples)
    split = {"method": "test", "budget_pct": "test", "seed": 0, "labelled_train": ids[:3],
             "labelled_val": ids[3:4], "unlabelled": ids[4:]}
    loc = Locations("local", fake_data, tmp_path / "out")
    loc.splits.mkdir(parents=True, exist_ok=True)
    (loc.splits / "unit_split.json").write_text(json.dumps(split), encoding="utf-8")
    for s in samples:
        if s["id"] in split["unlabelled"]:
            from pathlib import Path
            Path(s["mask"]).unlink(missing_ok=True)
    monkeypatch.setattr(semi, "pranet_train", lambda root: samples)
    cfg = load_config(None, {
        "name": name, "method": "mean_teacher",
        "model": {"name": "fulnet_c", "encoder": "mscan_t", "ual": "corrected", "pretrained": False},
        "data": {"img_size": 64, "split": "unit_split", "num_workers": 0},
        "fal": {"mode": "corrected", "beta": 0.05, "prob": 1.0},
        "loss": {"type": "structure", "mode": "corrected"},
        "ssl": {"unlabelled_batch": 2, "threshold": 0.5},
        "zoom": {"mode": zoom, "prob": 1.0},
        "train": {"iterations": iterations, "batch_size": 3, "lr": 1e-3, "amp": False, "grad_clip": 0.5,
                  "grad_clip_mode": "value", "ckpt_every": 2, "log_every": 2, "val_every": 2}})
    return cfg, loc


@pytest.mark.parametrize("zoom", ["off", "targeted", "fixed2"])
def test_mean_teacher_trains_without_reading_unlabelled_masks(fake_data, tmp_path, monkeypatch, zoom):
    from polypssl.engine import train
    from polypssl.engine.common import FINAL, RESUME
    cfg, loc = _semi_setup(fake_data, tmp_path, monkeypatch, zoom)
    run_dir = train(cfg, loc, torch.device("cpu"))
    log = pd.read_csv(run_dir / "log.csv")
    assert list(log["iteration"]) == [2, 4] and log["loss"].notna().all()
    assert (log["loss_unlabelled"] > 0).all() and log["val_dice"].notna().all()   # threshold 0.5: every pixel confident
    assert (log["zoomed_unlabelled"] > 0).all() == (zoom != "off")
    state = torch.load(run_dir / FINAL, weights_only=False)
    assert "student" in state and state["iteration"] == 4 and not (run_dir / RESUME).exists()
    # the saved model is the EMA teacher, which differs from the student after training
    assert any(not torch.equal(state["model"][k], state["student"][k]) for k in state["model"]
               if state["model"][k].dtype.is_floating_point)


def test_mean_teacher_resumes_after_disconnect(fake_data, tmp_path, monkeypatch):
    import polypssl.engine.semi as semi
    from polypssl.engine import train
    from polypssl.engine.common import FINAL
    cfg, loc = _semi_setup(fake_data, tmp_path, monkeypatch, iterations=6, name="mt_resume")
    real = semi.save_resume

    def save_then_disconnect(run_dir, payload):
        real(run_dir, payload)
        if payload["iteration"] == 4:
            raise KeyboardInterrupt
    monkeypatch.setattr(semi, "save_resume", save_then_disconnect)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, loc, torch.device("cpu"))
    monkeypatch.setattr(semi, "save_resume", real)
    run_dir = train(cfg, loc, torch.device("cpu"))
    assert list(pd.read_csv(run_dir / "log.csv")["iteration"]) == [2, 4, 6]
    assert (run_dir / FINAL).exists()


def test_zoom_requires_mean_teacher():
    with pytest.raises(ValueError):
        load_config(None, {"name": "x", "zoom": {"mode": "targeted"}})


# ---------------------------------------------------------------- Stage D selection
def _pool(n_kvasir=120, n_clinic=80):
    return ([{"id": f"cju{i:04d}", "source": "Kvasir"} for i in range(n_kvasir)]
            + [{"id": str(i + 1), "source": "ClinicDB"} for i in range(n_clinic)])


def test_acls_and_compmatched_splits():
    from polypssl.selection import acls_split, compmatched_split
    samples = _pool()
    feats = np.random.default_rng(0).normal(size=(len(samples), 5))
    acls = acls_split(samples, feats, "5", seed=0)
    lab = acls["labelled_train"] + acls["labelled_val"]
    assert len(lab) == 72 and len(acls["labelled_val"]) == 12 and len(acls["unlabelled"]) == 128
    assert acls == acls_split(samples, feats, "5", seed=0)                 # deterministic
    cm = compmatched_split(samples, acls, seed=0)
    src = {s["id"]: s["source"] for s in samples}
    count = lambda split, s: sum(src[i] == s for i in split["labelled_train"] + split["labelled_val"])
    assert all(count(cm, s) == count(acls, s) for s in ("Kvasir", "ClinicDB"))
    assert set(cm["labelled_train"] + cm["labelled_val"]) != set(lab)


def test_acls_picks_one_medoid_per_cluster():
    from polypssl.selection import acls_split
    samples = _pool(60, 60)
    rng = np.random.default_rng(1)
    centres = rng.normal(scale=50, size=(72, 4))
    feats = np.concatenate([centres + rng.normal(scale=0.1, size=centres.shape),
                            centres[:48] + rng.normal(scale=0.1, size=(48, 4))])
    split = acls_split(samples, feats, "5", seed=0)
    chosen = [i for i, s in enumerate(samples) if s["id"] in set(split["labelled_train"] + split["labelled_val"])]
    nearest = np.argmin(np.linalg.norm(feats[chosen][:, None] - centres[None], axis=2), axis=1)
    assert len(set(nearest.tolist())) == 72                                # every cluster represented once


def test_appearance_descriptor_shape_and_format_control():
    from polypssl.appearance import Standardiser, descriptor
    img = np.random.default_rng(0).uniform(0, 1, (300, 400, 3)).astype(np.float32)
    d = descriptor(img)
    assert d.shape == (57,) and np.isfinite(d).all()
    framed = img.copy(); framed[:, :40] = 0; framed[:, -40:] = 0           # black frame outside the centre
    assert np.allclose(descriptor(framed), d)                              # only the central 60 % is used
    z = Standardiser(np.stack([d, d * 1.1, d * 0.9]))(np.stack([d]))
    assert z.shape == (1, 57)
