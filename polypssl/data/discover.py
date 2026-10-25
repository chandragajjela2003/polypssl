"""Find image/mask pairs on disk for every dataset in the protocol.

Every sample is described by a small dict:
    {"image": path, "mask": path, "dataset": name, "source": hospital/source name, "id": stem}

Verified layouts (PraNet split, checked 2026-10-10 on the downloaded zips):
    TrainDataset/image/*.png  + TrainDataset/masks/*.png            (folder is "image", singular)
    TestDataset/<Name>/images/*.png + TestDataset/<Name>/masks/*.png
    Training file names: ClinicDB = numeric ("1.png" ... "612.png"), Kvasir = "cju..." style.

PolypGen and BKAI layouts are NOT yet verified (accounts pending). Their loaders check the
expected structure and stop with a clear message if it differs, instead of guessing silently.
"""
from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Test sets of the PraNet split, as named inside TestDataset/
PRANET_TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
PRANET_TEST_COUNTS = {
    "Kvasir": 100,
    "CVC-ClinicDB": 62,
    "CVC-ColonDB": 380,
    "CVC-300": 60,
    "ETIS-LaribPolypDB": 196,
}

# PolypGen centres used as unseen hospitals (C4 Oslo excluded: possible overlap with Kvasir-SEG)
POLYPGEN_CENTRES = ["C1", "C2", "C3", "C5", "C6"]


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def _find_mask(mask_dir: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask for '{stem}' in {mask_dir}")


def _pairs(image_dir: Path, mask_dir: Path, dataset: str, source_fn) -> list[dict]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected folders {image_dir} and {mask_dir}")

    def by_stem(folder: Path) -> dict[str, Path]:
        files = _list_images(folder)
        nested = [path for path in folder.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMG_EXTS and path.parent != folder]
        if nested:
            raise ValueError(f"Nested image files are not allowed in {folder}: {[p.name for p in nested[:5]]}")
        paths: dict[str, Path] = {}
        duplicates = []
        for path in files:
            if path.stem in paths:
                duplicates.append(path.stem)
            else:
                paths[path.stem] = path
        if duplicates:
            examples = ", ".join(sorted(set(duplicates))[:5])
            raise ValueError(f"Duplicate {dataset} file stems in {folder}: {examples}")
        return paths

    images, masks = by_stem(image_dir), by_stem(mask_dir)
    missing_masks = sorted(images.keys() - masks.keys())
    orphan_masks = sorted(masks.keys() - images.keys())
    if missing_masks or orphan_masks:
        details = []
        if missing_masks:
            details.append(f"{len(missing_masks)} images without masks (e.g. {missing_masks[:5]})")
        if orphan_masks:
            details.append(f"{len(orphan_masks)} masks without images (e.g. {orphan_masks[:5]})")
        raise ValueError(f"Invalid {dataset} image/mask pairing in {image_dir.parent}: {'; '.join(details)}")

    samples = []
    for stem, img in sorted(images.items()):
        samples.append({
            "image": str(img),
            "mask": str(masks[stem]),
            "dataset": dataset,
            "source": source_fn(stem),
            "id": stem,
        })
    if not samples:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return samples


def pranet_train(data_root: str | Path, expect_counts: bool = True) -> list[dict]:
    """The 1,450 training images: 900 Kvasir-SEG + 550 CVC-ClinicDB.
    expect_counts=False only for unit tests on small synthetic folders."""
    root = Path(data_root) / "TrainDataset"
    singular, plural = root / "image", root / "images"
    if singular.is_dir() and plural.is_dir():
        raise ValueError(f"Ambiguous training image folders: both {singular} and {plural} exist")
    image_dir = singular if singular.is_dir() else plural
    samples = _pairs(image_dir, root / "masks", "PraNetTrain",
                     lambda stem: "ClinicDB" if stem.isdigit() else "Kvasir")
    counts = {s: sum(1 for x in samples if x["source"] == s) for s in ("Kvasir", "ClinicDB")}
    if expect_counts and counts != {"Kvasir": 900, "ClinicDB": 550}:
        raise ValueError(f"Unexpected PraNet training composition {counts}; expected 900 Kvasir + 550 ClinicDB")
    return samples


def pranet_test(data_root: str | Path, name: str, expect_count: bool = True) -> list[dict]:
    if name not in PRANET_TEST_SETS:
        raise ValueError(f"Unknown PraNet test set '{name}'. Options: {PRANET_TEST_SETS}")
    root = Path(data_root) / "TestDataset" / name
    samples = _pairs(root / "images", root / "masks", name, lambda stem: name)
    expected = PRANET_TEST_COUNTS[name]
    if expect_count and len(samples) != expected:
        raise ValueError(f"Unexpected {name} count {len(samples)}; PraNet split requires {expected} pairs")
    return samples


def polypgen(data_root: str | Path, centre: str) -> list[dict]:
    """PolypGen single frames of one centre. Expected layout (to be confirmed after download):
    PolypGen/data_<centre>/images_<centre>/ and PolypGen/data_<centre>/masks_<centre>/"""
    root = Path(data_root) / "PolypGen" / f"data_{centre}"
    image_dir, mask_dir = root / f"images_{centre}", root / f"masks_{centre}"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"PolypGen layout not found at {image_dir}. The PolypGen loader must be checked against "
            f"the real download before use (layout not yet verified).")
    # PolypGen mask names carry a '_mask' suffix in the public release (to be verified)
    samples = []
    for img in _list_images(image_dir):
        try:
            mask = _find_mask(mask_dir, img.stem)
        except FileNotFoundError:
            mask = _find_mask(mask_dir, f"{img.stem}_mask")
        samples.append({"image": str(img), "mask": str(mask), "dataset": f"PolypGen-{centre}",
                        "source": f"PolypGen-{centre}", "id": img.stem})
    return samples


def bkai(data_root: str | Path) -> list[dict]:
    """BKAI-IGH NeoPolyp-Small training images with colour masks (red=neoplastic, green=non-neoplastic).
    Expected Kaggle layout (to be confirmed after download): BKAI/train/train/*.jpeg and
    BKAI/train_gt/train_gt/*.jpeg. Colour masks are merged to binary in the dataset class."""
    root = Path(data_root) / "BKAI"
    image_dir, mask_dir = root / "train" / "train", root / "train_gt" / "train_gt"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"BKAI layout not found at {image_dir}. The BKAI loader must be checked against the "
            f"real download before use (layout not yet verified).")
    return _pairs(image_dir, mask_dir, "BKAI", lambda stem: "BKAI")


def test_sets(data_root: str | Path, include_external: bool = True,
              expect_pranet_counts: bool = True) -> dict[str, list[dict]]:
    """All test sets. Block 1 = PraNet sets (FULNet comparison). Block 2 = external hospitals.
    External sets that are not downloaded yet are skipped with a printed note."""
    sets = {name: pranet_test(data_root, name, expect_pranet_counts) for name in PRANET_TEST_SETS}
    if include_external:
        for centre in POLYPGEN_CENTRES:
            try:
                sets[f"PolypGen-{centre}"] = polypgen(data_root, centre)
            except FileNotFoundError as err:
                print(f"[skip] PolypGen-{centre}: {err}")
        try:
            sets["BKAI"] = bkai(data_root)
        except FileNotFoundError as err:
            print(f"[skip] BKAI: {err}")
    return sets
