"""One entry point for every experiment in experiments.yaml.

    python run.py --list                    every run ID with its status for --seed (default 0)
    python run.py --run B2 --seed 0         train (resumes automatically) + evaluate; or run a diagnosis
    python run.py --results                 table of all finished runs (also saved as outputs/ALL_RESULTS.csv)
    python run.py --run B2 --iterations 20  smoke test: writes to outputs/_smoke/, never to the real folder
    python run.py --run all --reevaluate    evaluate every finished model again (e.g. after adding PolypGen)

Output layout (outputs = /content/outputs -> My Drive/polypssl/outputs on Colab):
    outputs/<ID>__<name>/seed<k>/   model_final.pth, log.csv, config.yaml, per_image.csv, summary.csv
                                    (checkpoint_resume.pth exists only while training is unfinished)
    outputs/ALL_RESULTS.csv          one row per run x seed x test set x size bin
Runs that already finished are never retrained. Folders of the old layout (outputs/runs/<name>/ and
outputs/results/<name>/) are moved into the new layout automatically.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from polypssl.config import load_config
from polypssl.engine.common import FINAL, LEGACY, RESUME, is_finished
from polypssl.evaluation import summarise
from polypssl.evaluation.evaluate import METRICS
from polypssl.paths import get_locations
from polypssl.utils import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "experiments.yaml"


def load_registry(path: Path = REGISTRY) -> dict:
    runs = yaml.safe_load(path.read_text(encoding="utf-8"))["runs"]
    for rid, e in runs.items():
        assert e["kind"] in {"train", "diagnosis"}, (rid, e["kind"])
        assert "__" not in e["name"], rid
        if e["kind"] == "train":
            assert (ROOT / e["config"]).is_file(), f"{rid}: missing {e['config']}"
        else:
            assert e["source"] in runs and runs[e["source"]]["kind"] == "train", rid
    return runs


def run_folder(out: Path, rid: str, entry: dict, seed: int) -> Path:
    return Path(out) / f"{rid}__{entry['name']}" / f"seed{seed}"


def _fill_seed(value, seed: int):
    if isinstance(value, dict):
        return {k: _fill_seed(v, seed) for k, v in value.items()}
    return value.replace("{seed}", str(seed)) if isinstance(value, str) else value


def build_config(entry: dict, seed: int, iterations: int | None = None) -> dict:
    overrides = _fill_seed(entry.get("overrides", {}), seed)
    overrides = {**overrides, "name": entry["name"], "seed": seed}
    if iterations is not None:
        overrides.setdefault("train", {})
        overrides["train"] = {**overrides["train"], "iterations": iterations}
    return load_config(ROOT / entry["config"], overrides)


# ---------------------------------------------------------------- status
def _last_logged(run_dir: Path) -> int | None:
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    try:
        df = pd.read_csv(log)
        return int(df["iteration"].iloc[-1]) if len(df) else None
    except Exception:
        return None


def status(out: Path, rid: str, entry: dict, seed: int, runs: dict) -> str:
    d = run_folder(out, rid, entry, seed)
    if entry.get("ready", True) is False:
        return "not ready: " + entry.get("note", "")
    if entry["kind"] == "diagnosis":
        if (d / "report.txt").exists():
            return "DONE"
        src = runs[entry["source"]]
        if not is_finished(run_folder(out, entry["source"], src, seed)):
            return f"waiting for {entry['source']}"
        return "partial (re-run to continue)" if d.exists() else "not started"
    if is_finished(d):
        return "DONE" if (d / "summary.csv").exists() else "trained, evaluation missing (re-run)"
    if (d / RESUME).exists() or (d / "log.csv").exists():
        it = _last_logged(d)
        total = yaml.safe_load((d / "config.yaml").read_text())["train"]["iterations"] \
            if (d / "config.yaml").exists() else "?"
        return f"partial: iteration {it}/{total} (re-run to resume)"
    return "not started"


def print_list(loc, runs: dict, seed: int) -> None:
    print(f"{'ID':4s} {'status (seed ' + str(seed) + ')':44s} what")
    for rid, e in runs.items():
        print(f"{rid:4s} {status(loc.out, rid, e, seed, runs)[:44]:44s} {e['what']}")
    print(f"\noutputs folder: {loc.out}")


# ---------------------------------------------------------------- old layout -> new layout
def migrate_legacy(loc, runs: dict) -> None:
    """outputs/runs/<legacy>/seed<k>/ and outputs/results/<legacy>/seed<k>/ -> outputs/<ID>__<name>/seed<k>/.
    A finished last.pth becomes model_final.pth (weights + config only); it is deleted only after the new file
    has been written and read back. An unfinished last.pth becomes checkpoint_resume.pth."""
    for rid, e in runs.items():
        old = e.get("legacy")
        if not old:
            continue
        for old_run in sorted((loc.runs / old).glob("seed*")) if (loc.runs / old).is_dir() else []:
            new = run_folder(loc.out, rid, e, int(old_run.name[4:]))
            new.mkdir(parents=True, exist_ok=True)
            print(f"[migrate] {old_run} -> {new}")
            for f in sorted(old_run.iterdir()):
                if f.name == LEGACY:
                    continue
                if not (new / f.name).exists():
                    shutil.move(str(f), str(new / f.name))
            last = old_run / LEGACY
            if last.exists():
                if is_finished(new):
                    print(f"[migrate] {new / FINAL} already exists; old {last} kept untouched")
                else:
                    state = load_checkpoint(last)
                    cfg = state["config"]
                    if state["iteration"] >= cfg["train"]["iterations"]:
                        save_checkpoint(new / FINAL, {"model": state["model"], "config": cfg,
                                                      "iteration": state["iteration"]})
                        check = torch.load(new / FINAL, map_location="cpu", weights_only=False)
                        assert check["iteration"] == state["iteration"] and \
                            check["model"].keys() == state["model"].keys()
                        last.unlink()
                        print(f"[migrate] finished checkpoint -> {new / FINAL} (verified); old last.pth deleted")
                    else:
                        shutil.move(str(last), str(new / RESUME))
                        print(f"[migrate] unfinished checkpoint -> {new / RESUME}")
            _remove_if_empty(old_run)
            _remove_if_empty(old_run.parent)
        for old_res in sorted((loc.results / old).glob("seed*")) if (loc.results / old).is_dir() else []:
            new = run_folder(loc.out, rid, e, int(old_res.name[4:]))
            new.mkdir(parents=True, exist_ok=True)
            for f in sorted(old_res.iterdir()):
                if not (new / f.name).exists():
                    shutil.move(str(f), str(new / f.name))
                    print(f"[migrate] {f} -> {new / f.name}")
            _remove_if_empty(old_res)
            _remove_if_empty(old_res.parent)


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------- results table
def collect_results(out: Path, runs: dict) -> pd.DataFrame:
    frames = []
    for rid, e in runs.items():
        if e["kind"] != "train":
            continue
        for summary in sorted((Path(out) / f"{rid}__{e['name']}").glob("seed*/summary.csv")):
            df = pd.read_csv(summary)
            per_image = summary.parent / "per_image.csv"
            if "miss_rate" not in df.columns and per_image.exists():     # older run: add the new columns
                pi = pd.read_csv(per_image, dtype={"id": str})            # (no GPU needed)
                if set(METRICS + ["size_bin", "empty_pred"]) <= set(pi.columns):
                    df = summarise(pi)
                    df.to_csv(summary, index=False)
            df.insert(0, "seed", int(summary.parent.name[4:]))
            df.insert(0, "name", e["name"])
            df.insert(0, "run_id", rid)
            frames.append(df)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not table.empty:
        table.to_csv(Path(out) / "ALL_RESULTS.csv", index=False)
    return table


def print_results(table: pd.DataFrame) -> None:
    if table.empty:
        print("No finished runs yet.")
        return
    pd.set_option("display.width", 200)
    for title, size_bin, value in (("Dice, all polyps", "all", "dice"),
                                   ("Dice, small polyps (< 2 % of the image)", "small", "dice"),
                                   ("Missed polyps (share of images with Dice < 0.1), all", "all", "miss_rate"),
                                   ("Missed polyps, small polyps only", "small", "miss_rate"),
                                   ("Dice on found polyps only (Dice >= 0.1)", "all", "dice_detected")):
        if value not in table.columns:
            continue
        t = table[table["size_bin"] == size_bin].pivot_table(index=["run_id", "seed"], columns="dataset",
                                                             values=value)
        print(f"\n{title}:\n{t.to_string(float_format='%.3f')}")


# ---------------------------------------------------------------- run
def run(rid: str, seed: int, iterations: int | None = None) -> Path:
    loc = get_locations()
    runs = load_registry()
    if rid not in runs:
        raise SystemExit(f"Unknown run ID {rid}. Known: {', '.join(runs)}")
    e = runs[rid]
    if e.get("ready", True) is False:
        raise SystemExit(f"{rid} is not ready yet: {e.get('note', '')}")
    migrate_legacy(loc, runs)
    smoke = iterations is not None
    out = loc.out / "_smoke" if smoke else loc.out
    d = run_folder(out, rid, e, seed)
    if smoke and d.exists():
        shutil.rmtree(d)                                   # smoke tests always start fresh
    print(f"[run] {rid} = {e['name']} seed {seed}: {e['what']}")
    print(f"[run] folder: {d}" + ("  (SMOKE TEST: not a real result)" if smoke else ""))
    print(f"[env] {loc.env} | data: {loc.data} | device: "
          f"{'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    if e["kind"] == "diagnosis":
        from polypssl.analysis.diagnosis import run_diagnosis
        src = run_folder(loc.out, e["source"], runs[e["source"]], seed)
        if not is_finished(src):
            raise SystemExit(f"{rid} needs the finished model of {e['source']} ({src / FINAL}); run that first")
        if (d / "report.txt").exists() and not smoke:
            print(f"[done] {d / 'report.txt'} already exists")
            print((d / "report.txt").read_text(encoding="utf-8"))
            return d
        run_diagnosis(src, d, loc, n_boot=200 if smoke else 2000)
        return d

    from polypssl.data import test_sets
    from polypssl.engine import train
    cfg = build_config(e, seed, iterations)
    test_sets(loc.data, include_external=False)
    print("[preflight] PraNet test folders, pairings and counts validated")
    train(cfg, loc, run_dir=d)
    if not (d / "summary.csv").exists():
        from evaluate import evaluate_run
        evaluate_run(d)
    if not smoke:
        table = collect_results(loc.out, runs)
        print(f"[saved] {loc.out / 'ALL_RESULTS.csv'} ({table['run_id'].nunique()} runs)")
    return d


def reevaluate(rid: str, seed: int) -> None:
    """Evaluate already-trained models again (e.g. after PolypGen is added); nothing is retrained.
    rid = one run ID, or 'all' for every finished training run of this seed."""
    from evaluate import evaluate_run
    loc = get_locations()
    runs = load_registry()
    migrate_legacy(loc, runs)
    ids = [r for r, e in runs.items() if e["kind"] == "train"] if rid.lower() == "all" else [rid]
    done = 0
    for r in ids:
        if r not in runs or runs[r]["kind"] != "train":
            raise SystemExit(f"--reevaluate needs a training run ID (or 'all'); got {r}")
        d = run_folder(loc.out, r, runs[r], seed)
        if not is_finished(d):
            if rid.lower() != "all":
                raise SystemExit(f"{r} seed {seed} has no model_final.pth yet; train it first")
            continue
        print(f"\n[reevaluate] {r} seed {seed}: {d}")
        evaluate_run(d)
        done += 1
    table = collect_results(loc.out, runs)
    print(f"[reevaluate] {done} model(s) evaluated again; {loc.out / 'ALL_RESULTS.csv'} updated "
          f"({table['run_id'].nunique() if not table.empty else 0} runs)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run ID from experiments.yaml, e.g. B2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--iterations", type=int, default=None, help="smoke test only (separate _smoke folder)")
    ap.add_argument("--reevaluate", action="store_true",
                    help="only evaluate the finished model of --run again (use --run all for every run); "
                         "e.g. after adding PolypGen. Nothing is retrained.")
    args = ap.parse_args(argv)
    loc = get_locations()
    runs = load_registry()
    if args.list:
        migrate_legacy(loc, runs)
        print_list(loc, runs, args.seed)
    if args.run and args.reevaluate:
        reevaluate(args.run, args.seed)
    elif args.run:
        run(args.run, args.seed, args.iterations)
    elif args.reevaluate:
        raise SystemExit("--reevaluate needs --run <ID> or --run all")
    if args.results:
        print_results(collect_results(loc.out, runs))
    if not (args.list or args.run or args.results):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
