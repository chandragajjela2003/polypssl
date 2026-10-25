"""Where data and outputs live, for each place the code can run.

Set POLYPSSL_ENV to colab, kaggle, or local for deterministic platform selection. If it is unset,
reliable runtime indicators are used. Any location can be overridden with environment variables:

    POLYPSSL_DATA   folder that contains the unzipped datasets (TrainDataset/, TestDataset/, ...)
    POLYPSSL_OUT    folder where splits/, runs/ and results/ are written

Default layout
--------------
Kaggle : data in /kaggle/working/data (unzipped from /kaggle/input/...), outputs in /kaggle/working/outputs
Colab  : data in /content/data (copied from Drive for speed), outputs in /content/outputs
Local  : data in ./data, outputs in ./outputs (relative to the current working directory)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def detect_environment() -> str:
    explicit = os.environ.get("POLYPSSL_ENV")
    if explicit:
        env = explicit.strip().lower()
        if env not in _DEFAULTS:
            raise ValueError(f"POLYPSSL_ENV must be one of {tuple(_DEFAULTS)}, got {explicit!r}")
        return env
    if any(key in os.environ for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")):
        return "kaggle"
    if "google.colab" in sys.modules or any(
        key in os.environ for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION")
    ):
        return "colab"
    return "local"


@dataclass(frozen=True)
class Locations:
    env: str
    data: Path
    out: Path

    @property
    def splits(self) -> Path:
        return self.out / "splits"

    @property
    def runs(self) -> Path:
        return self.out / "runs"

    @property
    def results(self) -> Path:
        return self.out / "results"

    @property
    def weights(self) -> Path:
        return self.out / "weights"


_DEFAULTS = {
    "kaggle": ("/kaggle/working/data", "/kaggle/working/outputs"),
    "colab": ("/content/data", "/content/outputs"),
    "local": ("data", "outputs"),
}


def get_locations(env: str | None = None) -> Locations:
    env = env or detect_environment()
    if env not in _DEFAULTS:
        raise ValueError(f"env must be one of {tuple(_DEFAULTS)}, got {env!r}")
    data_default, out_default = _DEFAULTS[env]
    data = Path(os.environ.get("POLYPSSL_DATA", data_default)).expanduser()
    out = Path(os.environ.get("POLYPSSL_OUT", out_default)).expanduser()
    return Locations(env=env, data=data, out=out)
