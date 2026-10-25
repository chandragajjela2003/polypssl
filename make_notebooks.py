"""Generate isolated Colab and Kaggle notebook pairs.

Run: python make_notebooks.py
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def session_start(platform):
    if platform == "colab":
        source = r'''
import os
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from google.colab import drive

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")

PLATFORM = "colab"
WORK = Path("/content")
CODE_DIR = WORK / "polypssl"
CODE_SOURCE = Path("/content/drive/MyDrive/polypssl/code/polypssl")
CODE_ZIP = Path("/content/drive/MyDrive/polypssl/code/polypssl.zip")

def extract_code_archive(archive, destination):
    with zipfile.ZipFile(archive) as zf:
        names = [PurePosixPath(info.filename.replace("\\", "/")) for info in zf.infolist() if not info.is_dir()]
        if any(path.is_absolute() or ".." in path.parts or
               (path.parts and path.parts[0].endswith(":")) for path in names):
            raise ValueError(f"Unsafe path found in {archive}")
        prefix = PurePosixPath("polypssl")
        if not any(path == prefix / "train.py" for path in names):
            prefix = PurePosixPath(".")
        if not any(path == prefix / "train.py" for path in names):
            raise FileNotFoundError("The code ZIP must contain train.py and the polypssl package")
        for info in zf.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if info.is_dir() or "__MACOSX" in path.parts:
                continue
            try:
                relative = path.relative_to(prefix) if str(prefix) != "." else path
            except ValueError:
                continue
            if not relative.parts:
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

if (CODE_SOURCE / "train.py").is_file() and (CODE_SOURCE / "polypssl").is_dir():
    shutil.copytree(CODE_SOURCE, CODE_DIR, dirs_exist_ok=True)
    print(f"Copied project folder from Drive: {CODE_SOURCE}")
elif CODE_ZIP.is_file():
    extract_code_archive(CODE_ZIP, CODE_DIR)
    print(f"Extracted project ZIP from Drive: {CODE_ZIP}")
else:
    raise FileNotFoundError(
        "Project not found. Upload the polypssl folder to My Drive/polypssl/code/polypssl "
        "or place polypssl.zip in My Drive/polypssl/code.")
os.environ["POLYPSSL_ENV"] = PLATFORM
os.environ["POLYPSSL_DATA"] = "/content/data"
OUTPUT_DIR = Path("/content/outputs")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/polypssl/outputs")
if not OUTPUT_DIR.exists() and not OUTPUT_DIR.is_symlink():
    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.symlink_to(DRIVE_OUTPUT_DIR, target_is_directory=True)
elif OUTPUT_DIR.is_symlink() and OUTPUT_DIR.resolve() != DRIVE_OUTPUT_DIR.resolve():
    raise RuntimeError(f"Refusing unexpected outputs symlink: {OUTPUT_DIR} -> {OUTPUT_DIR.resolve()}")
elif not OUTPUT_DIR.is_symlink():
    raise RuntimeError(f"{OUTPUT_DIR} is a normal folder, so results would NOT be saved to Drive. "
                       "Use Runtime > Disconnect and delete runtime, then run this notebook again.")
os.environ["POLYPSSL_OUT"] = str(OUTPUT_DIR)
'''
    else:
        source = r'''
import os
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

PLATFORM = "kaggle"
WORK = Path("/kaggle/working")
CODE_DIR = WORK / "polypssl"
INPUT_DIR = Path("/kaggle/input")
if not INPUT_DIR.is_dir():
    raise FileNotFoundError("This notebook must run in a Kaggle environment")

candidates = sorted({
    path.parent for path in INPUT_DIR.rglob("train.py")
    if (path.parent / "polypssl").is_dir()
})
if len(candidates) == 1:
    shutil.copytree(candidates[0], CODE_DIR, dirs_exist_ok=True)
elif len(candidates) > 1:
    raise RuntimeError(f"Multiple code roots found in Kaggle inputs: {candidates}")
else:
    archives = sorted(INPUT_DIR.rglob("polypssl.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(
            "Add exactly one code input with polypssl.zip or the extracted project folder")
    with zipfile.ZipFile(archives[0]) as zf:
        names = [PurePosixPath(info.filename.replace("\\", "/")) for info in zf.infolist() if not info.is_dir()]
        if any(path.is_absolute() or ".." in path.parts or
               (path.parts and path.parts[0].endswith(":")) for path in names):
            raise ValueError(f"Unsafe path found in {archives[0]}")
        prefix = PurePosixPath("polypssl")
        if not any(path == prefix / "train.py" for path in names):
            prefix = PurePosixPath(".")
        if not any(path == prefix / "train.py" for path in names):
            raise FileNotFoundError("The code ZIP must contain train.py and the polypssl package")
        for info in zf.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if info.is_dir() or "__MACOSX" in path.parts:
                continue
            try:
                relative = path.relative_to(prefix) if str(prefix) != "." else path
            except ValueError:
                continue
            if not relative.parts:
                continue
            target = CODE_DIR.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

os.environ["POLYPSSL_ENV"] = PLATFORM
os.environ["POLYPSSL_DATA"] = "/kaggle/working/data"
os.environ["POLYPSSL_OUT"] = "/kaggle/working/outputs"
'''
    return code(source + r'''
if not (CODE_DIR / "train.py").is_file():
    raise FileNotFoundError(f"train.py was not prepared in {CODE_DIR}")
os.chdir(CODE_DIR)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
print(f"[{PLATFORM}] code: {CODE_DIR}")
print(f"[{PLATFORM}] data: {os.environ['POLYPSSL_DATA']}")
print(f"[{PLATFORM}] outputs: {os.environ['POLYPSSL_OUT']}")
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')


def setup_notebook(platform):
    if platform == "colab":
        intro = r'''
# 00 · Colab setup and correctness checks

Keep this notebook under `notebooks/colab/`. In Google Drive, upload the project folder so the layout is:

- `My Drive/polypssl/code/polypssl/train.py`
- `My Drive/polypssl/code/polypssl/polypssl/`
- `My Drive/polypssl/data_zips/TrainDataset.zip` and `TestDataset.zip` (optional if setup downloads them)

The folder is copied to `/content/polypssl`; the notebook also accepts a ZIP as a fallback. Data goes to `/content/data`;
output paths are `/content/outputs` and new output folders are backed by `My Drive/polypssl/outputs` for checkpoint
persistence. Existing local output folders are preserved and reused. Re-running does not delete code or data, and
valid archives/checkpoints are reused.
Do not start training unless all data, weight, and test checks below pass.
'''
    else:
        intro = r'''
# 00 · Kaggle setup and correctness checks

Keep this notebook under `notebooks/kaggle/`. Add the project code as a Kaggle input dataset (either the extracted
project or its `polypssl.zip`), enable Internet for the one-time data/weight downloads, and select a GPU.
Code is prepared in `/kaggle/working/polypssl`; data and outputs are isolated under `/kaggle/working/`.
PolypGen and BKAI are optional external evaluation sets; neither is required for the core training experiment.
Do not start training unless all data, weight, and test checks below pass.
'''
    return notebook([
        md(intro),
        session_start(platform),
        md("## Install/check datasets and checkpoints\n\nThe setup inspects each archive and validates matching image/mask stems before extraction. It fails on missing, extra, or duplicate files. PolypGen and BKAI are optional; their external layouts are not part of the core preflight."),
        code('subprocess.run([sys.executable, "setup_env.py"], check=True)'),
        md("## Unit tests"),
        code(r'''
from polypssl.paths import get_locations
os.environ["POLYPSSL_TEST_WEIGHTS"] = str(get_locations().weights)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
'''),
        md("## Label splits"),
        code("!ls splits"),
        md("## Optional GPU speed checks"),
        code(r'''
!python speed_test.py --config configs/stageA/fulnet_c_mscan_full.yaml --iterations 50
!python speed_test.py --config configs/stageA/polyp_pvt_reproduce.yaml --iterations 50
!python speed_test.py --config configs/stageA/fulnet_c_pvt_full.yaml --iterations 50
'''),
    ])


def run_notebook(platform):
    if platform == "colab":
        where = """- Code: upload the whole `polypssl` folder to `My Drive/polypssl/code/polypssl/` (replace the old one).
- Results: `My Drive/polypssl/outputs/<ID>__<name>/seed<k>/` (the notebook links `/content/outputs` to it), so
  a disconnect loses at most 500 iterations: just run all cells again with the same `RUN_ID` and `SEED`."""
    else:
        where = """- Code: add the `polypssl` folder (or `polypssl.zip`) as a Kaggle input dataset; enable Internet and a GPU.
- Use **Save Version -> Save & Run All** so the run continues after you close the browser.
- Results: `/kaggle/working/outputs/<ID>__<name>/seed<k>/` = the notebook's Output tab. Download that folder
  (or copy it to `My Drive/polypssl/outputs/`) when the run finishes."""
    return notebook([
        md(f"""# Run one experiment ({platform})

1. Run the first two cells (they prepare code, data and pretrained weights; safe to re-run).
2. Set `RUN_ID` and `SEED` below (the list shows every ID and whether it is done), then run the rest.

{where}

Each run folder ends with `model_final.pth` (kept), `log.csv`, `config.yaml`, `per_image.csv`, `summary.csv`;
`outputs/ALL_RESULTS.csv` collects every finished run. Finished runs are never retrained.
"""),
        session_start(platform),
        code("!python setup_env.py"),
        code("!python run.py --list --seed 0"),
        md("## Choose the run"),
        code('RUN_ID = "C1"     # e.g. C1, B2, D1, E1  (see the list above and experiments.yaml)\nSEED = 0'),
        code("!python run.py --run {RUN_ID} --seed {SEED}"),
        md("## Results"),
        code(RESULTS_CELL),
        code("!python run.py --results"),
        md("## Only after a new test set (e.g. PolypGen) is added\n\nEvaluates every finished model again; "
           "nothing is retrained (a few minutes per model). Remove the `#` to run it."),
        code("# !python run.py --run all --reevaluate --seed {SEED}"),
    ])


RESULTS_CELL = """
from IPython.display import Image, display
from polypssl.paths import get_locations

folder = next(get_locations().out.glob(f"{RUN_ID}__*/seed{SEED}"), None)
print("run folder:", folder)
if folder is not None and (folder / "report.txt").exists():
    print((folder / "report.txt").read_text())
    for png in sorted(folder.glob("*.png")):
        display(Image(filename=str(png)))
"""


def main():
    root = Path(__file__).parent / "notebooks"
    for platform in ("colab", "kaggle"):
        folder = root / platform
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "00_setup_and_checks.ipynb").write_text(
            json.dumps(setup_notebook(platform), indent=1), encoding="utf-8")
        (folder / "run_experiment.ipynb").write_text(
            json.dumps(run_notebook(platform), indent=1), encoding="utf-8")
        (folder / "01_train_and_evaluate.ipynb").write_text(json.dumps(notebook([md(
            "# Replaced\n\nUse `run_experiment.ipynb` in this folder: set `RUN_ID` and `SEED`, then run all cells.")]),
            indent=1), encoding="utf-8")

    stubs = {
        "00_setup_and_checks.ipynb": "Setup notebooks are platform-specific. Open `colab/00_setup_and_checks.ipynb` or `kaggle/00_setup_and_checks.ipynb`.",
        "01_train_and_evaluate.ipynb": "Replaced. Open `colab/run_experiment.ipynb` or `kaggle/run_experiment.ipynb`.",
    }
    for filename, message in stubs.items():
        (root / filename).write_text(
            json.dumps(notebook([md(f"# Platform-specific notebook\n\n{message}")]), indent=1),
            encoding="utf-8")
    print("platform notebooks written to", root)


if __name__ == "__main__":
    main()
