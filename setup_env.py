"""Setup for a Colab or Kaggle session. Run it at the start of EVERY session; it is safe to re-run.

What it does
1. Installs the Python packages in requirements.txt (most are already present on Colab/Kaggle).
2. Makes the datasets available on the fast local disk (data folder):
     PraNet TrainDataset.zip / TestDataset.zip: if they are not already stored, they are downloaded ONCE from
     the PraNet authors' Google Drive links and kept (Colab: MyDrive/polypssl/data_zips, Kaggle:
     /kaggle/working/data_zips); each file's exact size is checked. Then they are unzipped to the data folder.
     Kaggle: dataset folders added as notebook inputs are linked instead (Kaggle auto-extracts uploads).
     Only KNOWN dataset zips are ever unzipped (never the code zip).
3. Downloads the MSCAN-T (FULNet) and PVTv2-B2 (Polyp-PVT) ImageNet weights, size-checked.
4. Prints what was found, so missing datasets are obvious.

Usage:  python setup_env.py            (add --skip-install to skip pip)
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from polypssl.data import PRANET_TEST_COUNTS, PRANET_TEST_SETS, pranet_test, pranet_train
from polypssl.paths import get_locations

MSCAN_URLS = [
    # GitHub API with raw media type (works where raw.githubusercontent.com is blocked)
    ("https://api.github.com/repos/zhaiying0925/FULNet/contents/pretrained/mscan_t.pth",
     {"Accept": "application/vnd.github.raw"}),
    ("https://github.com/zhaiying0925/FULNet/raw/main/pretrained/mscan_t.pth", {}),
]
MSCAN_BYTES = 15_833_551      # size of the released file (verified 2026-10-18)
PVT_URLS = [  # official PVTv2 ImageNet weights, github.com/whai362/PVT release v2 (asset 39461963)
    ("https://api.github.com/repos/whai362/PVT/releases/assets/39461963", {"Accept": "application/octet-stream"}),
    ("https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth", {}),
]
PVT_BYTES = 101_501_299       # verified 2026-10-18
EXPECTED = ["TrainDataset", "TestDataset"]
# PraNet split (github.com/DengPingFan/PraNet README): Google Drive file id and exact size (verified 2026-10-10)
PRANET_ZIPS = {
    "TrainDataset.zip": ("1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb", 418_892_179),
    "TestDataset.zip": ("1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao", 343_133_116),
}
DATASET_ZIPS = set(PRANET_ZIPS)   # only archives with verified layouts are automatically extracted


def _unzip(zip_path: Path, dest: Path):
    with zipfile.ZipFile(zip_path) as z:
        dataset = zip_path.stem
        files = []
        for member in z.infolist():
            if member.is_dir():
                continue
            relative_path = PurePosixPath(member.filename.replace("\\", "/"))
            parts = relative_path.parts
            if relative_path.is_absolute() or ".." in parts or (parts and parts[0].endswith(":")) or not parts:
                raise ValueError(f"Unsafe path in {zip_path.name}: {member.filename}")
            if "__MACOSX" in parts or any(part.startswith(".") for part in parts):
                continue
            try:
                root_index = parts.index(dataset)
            except ValueError:
                continue
            relative = PurePosixPath(*parts[root_index + 1:])
            if relative.parts:
                files.append((member, relative))

        _validate_zip_layout(zip_path, files)
        target_root = dest / dataset
        target_root.mkdir(parents=True, exist_ok=True)
        print(f"  unzip validated {zip_path.name} -> {target_root}")
        for member, relative in files:
            target = target_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _validate_zip_layout(zip_path: Path, files):
    dataset = zip_path.stem
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    members = {}
    for member, relative in files:
        name = str(relative)
        if name in members:
            raise ValueError(f"Duplicate archive path {name!r} in {zip_path.name}")
        members[name] = member
    if not members:
        raise ValueError(f"{zip_path.name} contains no files under its expected {dataset}/ folder")

    def check_pair(image_dir: str, mask_dir: str, expected: int | None = None) -> int:
        images = {}
        masks = {}
        for name in members:
            path = PurePosixPath(name)
            if path.suffix.lower() not in image_exts:
                continue
            if str(path.parent) == image_dir:
                target = images
            elif str(path.parent) == mask_dir:
                target = masks
            else:
                continue
            if path.stem in target:
                raise ValueError(f"Duplicate {dataset} image/mask stem {path.stem!r} in {zip_path.name}")
            target[path.stem] = name
        missing, orphan = images.keys() - masks.keys(), masks.keys() - images.keys()
        if missing or orphan:
            raise ValueError(
                f"Invalid {dataset} archive pairing in {zip_path.name}: "
                f"{len(missing)} images lack masks {list(sorted(missing))[:5]}, "
                f"{len(orphan)} masks lack images {list(sorted(orphan))[:5]}")
        if not images:
            raise ValueError(f"No images found at {image_dir} inside {zip_path.name}")
        if expected is not None and len(images) != expected:
            raise ValueError(
                f"{zip_path.name} has {len(images)} pairs under {image_dir}; expected {expected}")
        print(f"  [zip] {image_dir}: {len(images)} paired images/masks")
        return len(images)

    print(f"[zip] inspecting {zip_path.name} before extraction ({len(members)} files)")
    print(f"  [zip] first entries: {sorted(members)[:5]}")
    if dataset == "TrainDataset":
        image_dir = "image" if any(str(PurePosixPath(n).parent) == "image" for n in members) else "images"
        check_pair(image_dir, "masks", 1450)
        source_counts = {"Kvasir": 0, "ClinicDB": 0}
        for name in members:
            path = PurePosixPath(name)
            if str(path.parent) == image_dir and path.suffix.lower() in image_exts:
                source_counts["ClinicDB" if path.stem.isdigit() else "Kvasir"] += 1
        if source_counts != {"Kvasir": 900, "ClinicDB": 550}:
            raise ValueError(f"Unexpected PraNet training composition in {zip_path.name}: {source_counts}")
    elif dataset == "TestDataset":
        for name in PRANET_TEST_SETS:
            check_pair(f"{name}/images", f"{name}/masks", PRANET_TEST_COUNTS[name])
    else:
        raise ValueError(f"Refusing to extract an archive with an unverified layout: {zip_path.name}")


def zip_store(env: str, loc) -> Path:
    """Where dataset zips are kept between sessions."""
    if env == "colab":
        return Path("/content/drive/MyDrive/polypssl/data_zips")
    if env == "kaggle":
        return Path("/kaggle/working/data_zips")
    return loc.out / "data_zips"


def ensure_pranet_zips(store: Path):
    """Download the PraNet zips once (if not stored yet) and verify their exact sizes."""
    store.mkdir(parents=True, exist_ok=True)
    for name, (file_id, size) in PRANET_ZIPS.items():
        path = store / name
        if path.exists() and path.stat().st_size == size:
            print(f"[data] {name} present and verified ({size} bytes)")
            continue
        print(f"[data] downloading {name} ({size / 1e6:.0f} MB) from the PraNet Google Drive link ...")
        import gdown
        try:
            gdown.download(id=file_id, output=str(path), quiet=True)
        except Exception as err:
            print(f"[data] download error: {err}")
        if not path.exists() or path.stat().st_size != size:
            got = path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{name} could not be downloaded correctly ({got} of {size} bytes). Google Drive sometimes limits "
                f"popular files; wait an hour and re-run, or download it in a browser from "
                f"https://drive.google.com/file/d/{file_id}/view and upload it to {store}.")
        print(f"[data] {name} downloaded and verified")


def prepare_data(env: str, loc):
    data = loc.data
    data.mkdir(parents=True, exist_ok=True)
    if env == "kaggle":                                     # link dataset folders added as notebook inputs
        inputs = Path("/kaggle/input")
        bkai = inputs / "bkai-igh-neopolyp"                 # competition data, if added to the notebook
        if bkai.exists() and not (data / "BKAI").exists():
            os.symlink(bkai, data / "BKAI")
            print(f"  link {bkai} -> BKAI")
        for item in sorted(inputs.glob("*/**/*")):
            if item.is_file() and item.name in DATASET_ZIPS and not (data / item.stem).exists():
                _unzip(item, data)
            elif item.is_dir() and item.name in EXPECTED + ["PolypGen"] and not (data / item.name).exists():
                inner = item / item.name                    # Kaggle may extract X.zip into X/X/
                src = inner if inner.is_dir() else item
                os.symlink(src, data / item.name)           # read-only inputs are linked, not copied
                print(f"  link {src}")
    if all((data / name).exists() for name in EXPECTED):
        print(f"[data] PraNet data ready in {data}")
    elif env == "local":
        print(f"[data] local run: expecting datasets already unzipped in {data}")
        return
    else:
        store = zip_store(env, loc)
        ensure_pranet_zips(store)
        for z in sorted(store.glob("*.zip")):
            if z.name not in DATASET_ZIPS or (data / z.stem).exists():
                continue
            local = z
            if env == "colab":                              # unzip from local disk: much faster than Drive
                local = Path("/content") / z.name
                if not local.exists() or local.stat().st_size != z.stat().st_size:
                    print(f"  copy {z.name} from Drive to local disk")
                    shutil.copy2(z, local)
            _unzip(local, data)


POLYPGEN_ZIP = "PolypGen2021_MultiCenterData_v3.zip"          # Synapse syn45200214 (uploaded by the user)
POLYPGEN_EXPECTED = {"C1": 256, "C2": 301, "C3": 457, "C4": 227, "C5": 208, "C6": 88}   # paper; unverified


def extract_polypgen(zip_path: Path, data: Path) -> dict:
    """Extract ONLY the single-frame folders images_C<k>/ and masks_C<k>/ (video sequences are skipped) to
    data/PolypGen/data_C<k>/images_C<k>/ and .../masks_C<k>/, the layout polypssl/data/discover.py reads.
    The folder that contains them inside the zip does not matter; a centre whose folder appears in more
    than one place is skipped (ambiguous). Returns {centre: number of image/mask pairs}."""
    import re
    pattern = re.compile(r"^(images|masks)_(C[1-6])$")
    found = {}                                               # (kind, centre) -> {parent path: [members]}
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            path = PurePosixPath(m.filename.replace("\\", "/"))
            if m.is_dir() or "__MACOSX" in path.parts or path.name.startswith(".") or len(path.parts) < 2:
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in {zip_path.name}: {m.filename}")
            hit = pattern.match(path.parent.name)
            if hit and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                found.setdefault((hit.group(1), hit.group(2)), {}).setdefault(str(path.parent), []).append(m)
        counts = {}
        for centre in sorted({c for _, c in found}):
            imgs, msks = found.get(("images", centre), {}), found.get(("masks", centre), {})
            if len(imgs) != 1 or len(msks) != 1:
                print(f"  [PolypGen] {centre}: skipped (image folders {list(imgs)}, mask folders {list(msks)})")
                continue
            img_members, mask_members = next(iter(imgs.values())), next(iter(msks.values()))
            stems = {PurePosixPath(x.filename).stem for x in img_members}
            mask_stems = {PurePosixPath(x.filename).stem.removesuffix("_mask") for x in mask_members}
            if stems != mask_stems:
                print(f"  [PolypGen] {centre}: skipped ({len(stems - mask_stems)} images without mask, "
                      f"{len(mask_stems - stems)} masks without image)")
                continue
            root = data / "PolypGen" / f"data_{centre}"
            for kind, members in (("images", img_members), ("masks", mask_members)):
                out = root / f"{kind}_{centre}"
                out.mkdir(parents=True, exist_ok=True)
                for x in members:
                    with z.open(x) as src, (out / PurePosixPath(x.filename).name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            counts[centre] = len(stems)
            print(f"  [PolypGen] {centre}: {len(stems)} image/mask pairs "
                  f"(paper reports {POLYPGEN_EXPECTED.get(centre, '?')})")
    return counts


def prepare_polypgen(env: str, loc) -> None:
    """Optional external test set: never stops the setup."""
    if (loc.data / "PolypGen").exists():
        print(f"[data] PolypGen ready in {loc.data / 'PolypGen'}")
        return
    if env == "local":
        return
    z = zip_store(env, loc) / POLYPGEN_ZIP
    if not z.exists():
        print(f"[data] PolypGen not found ({z}); it is optional, evaluation will skip it")
        return
    try:
        local = z
        if env == "colab":                                   # unzip from local disk: much faster than Drive
            local = Path("/content") / z.name
            if not local.exists() or local.stat().st_size != z.stat().st_size:
                print(f"  copy {z.name} ({z.stat().st_size / 1e9:.1f} GB) from Drive to local disk")
                shutil.copy2(z, local)
        print(f"[data] extracting PolypGen single frames from {z.name}")
        counts = extract_polypgen(local, loc.data)
        if not counts:
            shutil.rmtree(loc.data / "PolypGen", ignore_errors=True)
            print("[data] PolypGen: no usable centre found; send the zip listing for a loader fix")
    except Exception as err:
        shutil.rmtree(loc.data / "PolypGen", ignore_errors=True)
        print(f"[data] PolypGen could not be prepared ({err}); evaluation will skip it")


def download_weights(weights: Path):
    _download(weights / "mscan_t.pth", MSCAN_URLS, MSCAN_BYTES)     # FULNet MSCAN-T encoder
    _download(weights / "pvt_v2_b2.pth", PVT_URLS, PVT_BYTES)       # Polyp-PVT backbone


def _download(target: Path, urls, expected_bytes: int):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_bytes:
        print(f"[weights] {target.name} present and verified ({expected_bytes} bytes)")
        return
    for url, headers in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r, open(target, "wb") as f:
                shutil.copyfileobj(r, f)
            if target.stat().st_size == expected_bytes:
                print(f"[weights] downloaded and verified {target.name} from {url.split('/')[2]}")
                return
            print(f"[weights] wrong size from {url} ({target.stat().st_size} bytes), trying next source")
        except Exception as err:                       # network blocked, rate limit, ...
            print(f"[weights] failed from {url}: {err}")
    target.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {target.name}. Upload it manually to {target.parent}.")


def report(data: Path):
    print("[check] datasets found:")
    for name in ["TrainDataset", "TestDataset", "PolypGen", "BKAI"]:
        p = data / name
        n = sum(1 for _ in p.rglob("*") if _.suffix.lower() in {".png", ".jpg", ".jpeg"}) if p.exists() else 0
        status = "optional" if name in {"PolypGen", "BKAI"} and not p.exists() else (
            "present" if p.exists() else "MISSING")
        print(f"  {name:13s} {status:8s} {n} image/mask files")
    print("[check] PolypGen/BKAI are optional; their layouts are not included in the core preflight.")


def validate_core_datasets(data: Path):
    import cv2

    train_samples = pranet_train(data)
    counts = {
        source: sum(sample["source"] == source for sample in train_samples)
        for source in ("Kvasir", "ClinicDB")
    }
    print(f"[check] TrainDataset: {len(train_samples)} valid pairs ({counts})")
    test_samples = {name: pranet_test(data, name) for name in PRANET_TEST_SETS}
    test_counts = {name: len(samples) for name, samples in test_samples.items()}
    print(f"[check] TestDataset: {sum(test_counts.values())} valid pairs ({test_counts})")
    for sample in train_samples + [item for samples in test_samples.values() for item in samples]:
        image = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(sample["mask"], cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Unreadable image/mask file for sample {sample['id']}: "
                             f"{sample['image']} | {sample['mask']}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ for sample {sample['id']}: "
                             f"{image.shape[:2]} vs {mask.shape[:2]}")
    print("[check] all core image/mask files decode and have matching dimensions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()
    loc = get_locations()
    print(f"[env] {loc.env} | data -> {loc.data} | outputs -> {loc.out}")
    if not args.skip_install:
        req = Path(__file__).parent / "requirements.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    prepare_data(loc.env, loc)
    validate_core_datasets(loc.data)
    prepare_polypgen(loc.env, loc)
    download_weights(loc.weights)
    report(loc.data)
    print("[ready] core PraNet train/test datasets and both model checkpoints are validated")


if __name__ == "__main__":
    main()
