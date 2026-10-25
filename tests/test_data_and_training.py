"""Data pipeline alignment, end-to-end training (loss goes down), resume after interruption, evaluation."""
import random

import numpy as np
import pandas as pd
import pytest
import torch

from polypssl.config import load_config
from polypssl.data import LabelledDataset, pranet_test, pranet_train
from polypssl.data.transforms import load_image, load_mask, train_transform
from polypssl.paths import Locations


def test_discovery_reads_singular_image_folder_and_sources(fake_data):
    samples = pranet_train(fake_data, expect_counts=False)
    assert len(samples) == 10
    assert sum(s["source"] == "ClinicDB" for s in samples) == 4
    assert len(pranet_test(fake_data, "ETIS-LaribPolypDB", expect_count=False)) == 2


def test_discovery_rejects_orphan_masks(fake_data):
    from pathlib import Path

    (Path(fake_data) / "TrainDataset" / "masks" / "orphan.png").write_bytes(b"")
    with pytest.raises(ValueError, match="masks without images"):
        pranet_train(fake_data, expect_counts=False)


def test_discovery_rejects_duplicate_stems(fake_data):
    from pathlib import Path

    (Path(fake_data) / "TrainDataset" / "masks" / "1.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="Duplicate PraNetTrain file stems"):
        pranet_train(fake_data, expect_counts=False)


def test_discovery_rejects_mixed_image_folder_names(fake_data):
    from pathlib import Path

    (Path(fake_data) / "TrainDataset" / "images").mkdir()
    with pytest.raises(ValueError, match="Ambiguous training image folders"):
        pranet_train(fake_data, expect_counts=False)


def test_augmentation_keeps_image_and_mask_aligned(fake_data):
    s = pranet_train(fake_data, expect_counts=False)[0]
    img, mask = load_image(s["image"]), load_mask(s["mask"])
    for k in range(50):
        a_img, a_mask = train_transform(img, mask, 64, random.Random(k))
        inside = a_img.mean(axis=2)[a_mask == 1].mean()
        outside = a_img.mean(axis=2)[a_mask == 0].mean()
        assert set(np.unique(a_mask)) <= {0, 1}
        assert inside > outside + 0.3          # bright square stays where the mask says


def test_dataset_tensors(fake_data):
    ds = LabelledDataset(pranet_train(fake_data, expect_counts=False), img_size=64, augment=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)
    assert 0 <= x.min() and x.max() <= 1 and set(y.unique().tolist()) <= {0.0, 1.0}


def _cfg(tmp_path, iterations, name="smoke"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"name: {name}\n"
        "model: {name: fulnet_c, encoder: mscan_t, ual: corrected, pretrained: false}\n"
        "data: {img_size: 64, split: full, num_workers: 0}\n"
        "fal: {mode: corrected, beta: 0.05, prob: 0.5}\n"
        "loss: {mode: corrected}\n"
        f"train: {{iterations: {iterations}, batch_size: 4, lr: 1.0e-3, amp: false, "
        "ckpt_every: 5, log_every: 5, val_every: 1000}\n")
    return load_config(path)


def _patch_train_pool(monkeypatch, fake_data):
    import polypssl.engine.supervised as sup
    monkeypatch.setattr(sup, "pranet_train", lambda root: pranet_train(root, expect_counts=False))


def test_training_reduces_loss(fake_data, tmp_path, monkeypatch):
    from polypssl.engine import train
    _patch_train_pool(monkeypatch, fake_data)
    loc = Locations("local", fake_data, tmp_path / "out")
    run_dir = train(_cfg(tmp_path, 60), loc, torch.device("cpu"))
    log = pd.read_csv(run_dir / "log.csv")
    assert log["loss"].iloc[-3:].mean() < 0.8 * log["loss"].iloc[:2].mean()


def test_resume_continues_from_checkpoint(fake_data, tmp_path, monkeypatch):
    """Simulated disconnect right after the iteration-10 checkpoint; re-running resumes from 10, then the
    finished run keeps only model_final.pth (the resume checkpoint is deleted)."""
    from polypssl.engine import common, train
    import polypssl.engine.supervised as sup
    _patch_train_pool(monkeypatch, fake_data)
    loc = Locations("local", fake_data, tmp_path / "out")
    real_save = common.save_resume

    def save_then_disconnect(run_dir, payload):
        real_save(run_dir, payload)
        if payload["iteration"] == 10:
            raise KeyboardInterrupt("simulated disconnect")
    monkeypatch.setattr(sup, "save_resume", save_then_disconnect)
    with pytest.raises(KeyboardInterrupt):
        train(_cfg(tmp_path, 15, "resume"), loc, torch.device("cpu"))
    run_dir = loc.runs / "resume" / "seed0"
    assert torch.load(run_dir / common.RESUME, weights_only=False)["iteration"] == 10
    monkeypatch.setattr(sup, "save_resume", real_save)
    train(_cfg(tmp_path, 15, "resume"), loc, torch.device("cpu"))
    log = pd.read_csv(run_dir / "log.csv")
    assert list(log["iteration"]) == [5, 10, 15]                               # no repeated iterations
    assert (run_dir / common.FINAL).exists() and not (run_dir / common.RESUME).exists()
    assert torch.load(run_dir / common.FINAL, weights_only=False)["iteration"] == 15
    train(_cfg(tmp_path, 15, "resume"), loc, torch.device("cpu"))              # finished: skipped
    assert list(pd.read_csv(run_dir / "log.csv")["iteration"]) == [5, 10, 15]


def test_resume_trims_log_rows_after_checkpoint(tmp_path):
    from polypssl.engine.common import trim_log
    (tmp_path / "log.csv").write_text("iteration,loss\n5,1\n10,1\n15,1\n", encoding="utf-8")
    trim_log(tmp_path, 10)
    assert (tmp_path / "log.csv").read_text(encoding="utf-8") == "iteration,loss\n5,1\n10,1\n"


def test_evaluation_writes_per_image_and_summary(fake_data, tmp_path):
    from polypssl.data import test_sets
    from polypssl.evaluation import evaluate_model, save_results
    from polypssl.models.fulnet import FULNetC
    per_image = evaluate_model(FULNetC(ual="corrected"),
                               test_sets(fake_data, include_external=False, expect_pranet_counts=False),
                               64, torch.device("cpu"), batch_size=2, num_workers=0, amp=False)
    summary = save_results(per_image, tmp_path / "res")
    assert len(per_image) == 10 and set(per_image["dataset"]) == {"Kvasir", "CVC-ClinicDB", "CVC-ColonDB",
                                                                   "CVC-300", "ETIS-LaribPolypDB"}
    assert (tmp_path / "res" / "per_image.csv").exists() and "size_bin" in summary.columns
    # metrics are computed at the ORIGINAL mask size (80 x 100), not the network size (64)
    assert set(per_image["height"]) == {80} and set(per_image["width"]) == {100}


def test_polyp_pvt_recipe_trains(fake_data, tmp_path, monkeypatch):
    """Polyp-PVT recipe: no flips, multi-scale steps (96/128/160 px at size 128), value clipping, FAL + UAL."""
    from polypssl.engine import train
    _patch_train_pool(monkeypatch, fake_data)
    path = tmp_path / "pvt.yaml"
    path.write_text(
        "name: pvt_smoke\n"
        "model: {name: polyp_pvt, ual: corrected, pretrained: false}\n"
        "data: {img_size: 128, split: full, augment: none, num_workers: 0}\n"
        "fal: {mode: corrected, beta: 0.02, prob: 1.0}\n"
        "loss: {type: structure, mode: corrected}\n"
        "train: {iterations: 4, batch_size: 4, lr: 1.0e-4, amp: false, grad_clip: 0.5, grad_clip_mode: value, "
        "multiscale: [0.75, 1.0, 1.25], ckpt_every: 2, log_every: 2, val_every: 1000}\n")
    loc = Locations("local", fake_data, tmp_path / "out")
    run_dir = train(load_config(path), loc, torch.device("cpu"))
    log = pd.read_csv(run_dir / "log.csv")
    assert len(log) == 2 and log["loss"].notna().all() and (log["loss_fal"] > 0).all()
