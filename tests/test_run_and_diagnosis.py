"""run.py (registry, fairness of the B/D/E settings, old-layout migration, results table) and the C1 diagnosis."""
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import run as runner
from polypssl.analysis import diagnosis as dg
from polypssl.config import load_config
from polypssl.engine.common import FINAL, RESUME, finalize
from polypssl.paths import Locations


# ---------------------------------------------------------------- registry
def test_registry_entries_build_and_their_splits_exist():
    runs = runner.load_registry()
    for rid, e in runs.items():
        if e["kind"] != "train" or e.get("ready", True) is False:
            continue
        for seed in (0, 1, 2):
            cfg = runner.build_config(e, seed)
            assert cfg["seed"] == seed and cfg["name"] == e["name"]
            split = cfg["data"]["split"]
            assert split == "full" or (runner.ROOT / "splits" / f"{split}.json").is_file(), (rid, split)
            if split != "full":
                assert split.endswith(f"seed{seed}")


def test_b_d_e_runs_differ_only_in_method_zoom_and_split():
    """Fair comparison: every 5 % run uses the identical network, data recipe, optimiser and schedule."""
    runs = runner.load_registry()
    ids = ["B1", "B2", "D1", "D2", "E1", "E2"]
    cfgs = {rid: runner.build_config(runs[rid], 0) for rid in ids}
    strip = lambda c: {k: v for k, v in c.items() if k not in {"name", "method", "zoom"}} | {
        "data": {k: v for k, v in c["data"].items() if k != "split"}}
    ref = strip(cfgs["B1"])
    for rid in ids:
        assert strip(cfgs[rid]) == ref, rid
    assert cfgs["B1"]["method"] == "supervised" and all(cfgs[r]["method"] == "mean_teacher" for r in ids[1:])
    assert cfgs["E1"]["zoom"]["mode"] == "targeted" and cfgs["E2"]["zoom"]["mode"] == "fixed2"
    assert all(cfgs[r]["zoom"]["mode"] == "off" for r in ("B2", "D1", "D2"))
    assert cfgs["B1"]["train"]["iterations"] == 17820 and cfgs["B1"]["data"]["img_size"] == 320
    assert cfgs["B2"]["data"]["split"] == cfgs["E1"]["data"]["split"] == "random_5pct_seed0"


def test_not_ready_run_is_refused():
    with pytest.raises(SystemExit):
        runner.run("D3", 0)


# ---------------------------------------------------------------- migration and results table
def _legacy_run(out, iterations_done):
    cfg = load_config(None, {"name": "polyp_pvt_reproduce", "train": {"iterations": 10}})
    old = out / "runs" / "polyp_pvt_reproduce" / "seed0"
    old.mkdir(parents=True)
    torch.save({"model": {"w": torch.ones(3)}, "optim": {}, "config": cfg, "iteration": iterations_done},
               old / "last.pth")
    (old / "log.csv").write_text("iteration,loss\n10,0.1\n", encoding="utf-8")
    res = out / "results" / "polyp_pvt_reproduce" / "seed0"
    res.mkdir(parents=True)
    (res / "summary.csv").write_text("dataset,size_bin,dice\nKvasir,all,0.9\n", encoding="utf-8")
    (res / "per_image.csv").write_text("dataset,id,dice\nKvasir,a,0.9\n", encoding="utf-8")
    return Locations("local", out / "data", out)


def test_migration_of_finished_legacy_run(tmp_path):
    loc = _legacy_run(tmp_path, 10)
    runs = runner.load_registry()
    runner.migrate_legacy(loc, runs)
    new = tmp_path / "A1__polyp_pvt_reproduce" / "seed0"
    assert {p.name for p in new.iterdir()} == {FINAL, "log.csv", "summary.csv", "per_image.csv"}
    state = torch.load(new / FINAL, weights_only=False)
    assert state["iteration"] == 10 and "optim" not in state and torch.equal(state["model"]["w"], torch.ones(3))
    assert not (tmp_path / "runs" / "polyp_pvt_reproduce").exists()
    assert not (tmp_path / "results" / "polyp_pvt_reproduce").exists()
    runner.migrate_legacy(loc, runs)                                       # idempotent
    assert runner.status(tmp_path, "A1", runs["A1"], 0, runs) == "DONE"
    table = runner.collect_results(tmp_path, runs)
    assert list(table["run_id"]) == ["A1"] and (tmp_path / "ALL_RESULTS.csv").exists()


def test_migration_of_unfinished_legacy_run(tmp_path):
    loc = _legacy_run(tmp_path, 5)
    runner.migrate_legacy(loc, runner.load_registry())
    new = tmp_path / "A1__polyp_pvt_reproduce" / "seed0"
    assert (new / RESUME).exists() and not (new / FINAL).exists()


# ---------------------------------------------------------------- diagnosis maths
def test_oaxaca_parts_sum_to_gap_and_isolate_size():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b_in = rng.choice(dg.BINS, 100, p=[0.1, 0.5, 0.4])
        b_un = rng.choice(dg.BINS, 80, p=[0.5, 0.3, 0.2])
        d_in, d_un = rng.uniform(size=100), rng.uniform(size=80)
        gap, size_part, within = dg.oaxaca(d_in, b_in, d_un, b_un)
        assert gap == pytest.approx(size_part + within)
    # same Dice per bin in both groups -> the whole gap is size composition
    level = {"small": 0.6, "medium": 0.85, "large": 0.95}
    b_in = np.array(["small"] * 10 + ["medium"] * 50 + ["large"] * 40)
    b_un = np.array(["small"] * 50 + ["medium"] * 30 + ["large"] * 20)
    d_in, d_un = np.array([level[b] for b in b_in]), np.array([level[b] for b in b_un])
    gap, size_part, within = dg.oaxaca(d_in, b_in, d_un, b_un)
    assert gap > 0.05 and size_part == pytest.approx(gap) and within == pytest.approx(0.0, abs=1e-12)


def test_partial_spearman_removes_shared_factor():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.1 * rng.normal(size=500), z + 0.1 * rng.normal(size=500)
    assert abs(dg.partial_spearman(x, y, z)[0]) < 0.2 and np.corrcoef(x, y)[0, 1] > 0.9
    assert np.isnan(dg.partial_spearman(np.zeros(50), y[:50], z[:50])[0])   # constant Dice -> undefined


def _photo(seed=0, size=64):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.uniform(0.2, 0.8, (size, size, 3)) * np.array([1.0, 0.6, 0.5]), 0, 1).astype(np.float32)
    img[:, :4] = 0                                                          # black border
    return img


def test_normalisers():
    img, other = _photo(0), _photo(1)
    target = dg._lab_stats(other)
    out = dg.Reinhard(*target)(img)
    m, s = dg._lab_stats(out)
    assert np.allclose(m, target[0], atol=1.5) and np.allclose(s, target[1], rtol=0.1, atol=1.0)
    assert np.all(out[:, :4] == 0)                                          # border untouched
    valid = dg._valid(img)
    q = np.clip(np.round(img * 255), 0, 255).astype(np.int64)
    cdf = np.stack([np.cumsum(np.bincount(q[..., c][valid], minlength=256)) / valid.sum() for c in range(3)])
    assert np.abs(dg.HistMatch(cdf)(img) - img).max() <= 1 / 255 + 1e-6     # own histogram -> unchanged
    amp = np.abs(np.fft.fft2(img.transpose(2, 0, 1).astype(np.float64)))
    assert np.abs(dg.LowFreqAmplitude(amp, 0.05)(img) - img).max() < 1e-5   # own amplitude -> unchanged
    assert dg.Gamma(5)(img).mean() < img.mean() and dg.ColourTemperature(3)(img)[..., 0].mean() > img[..., 0].mean()


def test_diagnosis_end_to_end(fake_data, tmp_path, monkeypatch):
    from polypssl.data import pranet_train, test_sets
    from polypssl.models import build_model
    monkeypatch.setattr(dg, "pranet_train", lambda root: pranet_train(root, expect_counts=False))
    monkeypatch.setattr(dg, "test_sets", lambda root, ext: test_sets(root, False, expect_pranet_counts=False))
    cfg = load_config(None, {"name": "src", "model": {"name": "fulnet_c", "pretrained": False},
                             "data": {"img_size": 64, "num_workers": 0}, "train": {"iterations": 1, "amp": False}})
    src = tmp_path / "src"
    src.mkdir()
    finalize(src, build_model(cfg), cfg, 1)
    out = dg.run_diagnosis(src, tmp_path / "C1", Locations("local", fake_data, tmp_path), torch.device("cpu"),
                           include_external=False, n_boot=20)
    for f in ("per_image.csv", "decomposition.csv", "misses.csv", "normalisers.csv", "controls.csv", "asi.csv", "report.txt",
              "dice_vs_factors.png", "gap_decomposition.png"):
        assert (out / f).exists(), f
    per_image = pd.read_csv(out / "per_image.csv")
    assert len(per_image) == 10 and per_image["appearance_dist"].notna().all()
    dec = pd.read_csv(out / "decomposition.csv").dropna(subset=["gap"])
    assert np.allclose(dec["gap"], dec["size_part"] + dec["within_part"])
    assert len(pd.read_csv(out / "asi.csv")) == 14
    # cached passes make a re-run continue instead of recomputing
    n_passes = len(list((out / "passes").glob("*.csv")))
    dg.run_diagnosis(src, out, Locations("local", fake_data, tmp_path), torch.device("cpu"),
                     include_external=False, n_boot=20)
    assert len(list((out / "passes").glob("*.csv"))) == n_passes


# ---------------------------------------------------------------- missed polyps and re-evaluation
def _per_image(dices, bins, dataset):
    return pd.DataFrame({"dataset": dataset, "id": [str(i) for i in range(len(dices))], "dice": dices,
                         "iou": dices, "precision": dices, "recall": dices, "hd95_pct": 1.0, "mdice_thr": dices,
                         "empty_pred": False, "size_bin": bins, "area_pct": 1.0})


def test_summary_has_miss_rate_and_detected_dice():
    from polypssl.evaluation import summarise
    s = summarise(_per_image([0.0, 0.05, 0.8, 0.9], ["small", "small", "large", "large"], "ETIS-LaribPolypDB"))
    row = s[s["size_bin"] == "all"].iloc[0]
    assert row["miss_rate"] == 0.5 and row["dice_detected"] == pytest.approx(0.85)
    small = s[s["size_bin"] == "small"].iloc[0]
    assert small["miss_rate"] == 1.0 and np.isnan(small["dice_detected"])


def test_miss_analysis_attributes_gap_to_extra_misses():
    ind = _per_image([0.9] * 10, ["medium"] * 10, "Kvasir")
    uns = _per_image([0.9] * 8 + [0.0] * 2, ["small"] * 2 + ["medium"] * 6 + ["small"] * 2, "ETIS-LaribPolypDB")
    m = dg.miss_analysis(pd.concat([ind, uns]), {"ETIS": ["ETIS-LaribPolypDB"]}).set_index("group")
    assert m.loc["ETIS", "gap"] == pytest.approx(0.18) and m.loc["ETIS", "extra_miss_part"] == pytest.approx(0.18)
    assert m.loc["ETIS", "miss_rate_small"] == 0.5 and m.loc["ETIS", "miss_rate_not_small"] == 0.0


def test_old_summaries_gain_miss_columns_without_gpu(tmp_path):
    runs = runner.load_registry()
    d = tmp_path / "B2__pvtfu_mt_random5" / "seed0"
    d.mkdir(parents=True)
    _per_image([0.0, 0.9], ["small", "large"], "Kvasir").to_csv(d / "per_image.csv", index=False)
    pd.DataFrame({"dataset": ["Kvasir"], "size_bin": ["all"], "dice": [0.45]}).to_csv(d / "summary.csv", index=False)
    table = runner.collect_results(tmp_path, runs)
    assert "miss_rate" in table.columns and "miss_rate" in pd.read_csv(d / "summary.csv").columns


def test_reevaluate_refuses_unfinished_or_analysis_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYPSSL_ENV", "local")
    monkeypatch.setenv("POLYPSSL_OUT", str(tmp_path))
    with pytest.raises(SystemExit):
        runner.reevaluate("B2", 0)                                         # not trained
    with pytest.raises(SystemExit):
        runner.reevaluate("C1", 0)                                         # an analysis, not a model
    runner.reevaluate("all", 0)                                            # nothing trained: does nothing
