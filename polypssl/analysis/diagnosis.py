"""C1 / C2 diagnosis: WHY does a trained model lose Dice on unseen hospitals? (NOVELTY.md v7, P0 / C1)

Inputs: one finished run (C1 = A1, Polyp-PVT at 100 % labels; C2 = B2, Mean Teacher at 5 %). Nothing is
trained and nothing is tuned; every setting below is fixed in advance.

1. Per-image factors. For every test image: Dice (threshold 0.5, native resolution), polyp area (% of the
   image, from the ground-truth mask) and APPEARANCE DISTANCE = mean Euclidean distance to the 10 nearest
   training images in the standardised 57-number appearance descriptor space (appearance.py; images only).
   Spearman correlation of Dice with distance and with area, per dataset, plus PARTIAL correlations
   (each factor with the other held fixed, rank-based).
2. Gap decomposition (Oaxaca-Blinder over polyp-size bins small < 2 %, medium 2-10 %, large > 10 %).
   In-domain reference = Kvasir + CVC-ClinicDB test images. For an unseen set, with w_b = share of images
   in bin b and D_b = mean Dice in bin b:
        G = D_in - D_un = sum_b (w_in,b - w_un,b) * D_in,b   +   sum_b w_un,b * (D_in,b - D_un,b)
                          ^ size-composition part                ^ within-size part (appearance, other)
   95 % bootstrap intervals (2,000 resamples of images within each group).
3. Appearance normalisers applied to the unseen test images (network input, after resizing), each
   moving the image towards the TRAINING images' statistics (images only):
        reinhard   : per-channel mean/std in CIELAB matched to the training average (Reinhard 2001)
        histmatch  : per-channel histogram matched to the pooled training histogram
        lowfreq    : low-frequency Fourier amplitude (window floor(size * 0.01)) replaced by the training
                     mean amplitude (FDA-style, as in FAL corrected)
   Black endoscope borders (max channel <= 0.05) are left untouched by the first two.
   "Dice recovered" = normalised - original Dice. A normaliser removes only part of any appearance shift,
   so recovered Dice is a LOWER BOUND on the appearance-caused loss.
4. Controls. Negative: the same normalisers on in-domain images (should change Dice little).
   Positive: in-domain images shifted to ETIS colour statistics (Reinhard towards ETIS), then normalised
   (shows the normalisers CAN recover an appearance-caused drop when one exists).
5. Appearance sensitivity (dose-response) on in-domain images: gamma (darkening) and colour temperature
   (warmer), 5 severities each. Normalised AUC = mean_s Dice_s / Dice_0; ASI = 1 - normalised AUC.

6. Missed polyps (added 2026-10-22 after C1): share of images with Dice < 0.1 (overall, small polyps,
   others), Dice of the found polyps, and the part of each gap caused by extra misses.

Outputs (out_dir): per_image.csv, correlations.csv, decomposition.csv, misses.csv, normalisers.csv, controls.csv,
asi.csv, figures (*.png) and report.txt. Intermediate passes are cached in out_dir/passes/, so an
interrupted diagnosis continues where it stopped.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ..appearance import Standardiser, descriptor
from ..data import pranet_train, test_sets
from ..data.transforms import load_image
from ..engine.common import build_trained_model
from ..evaluation import evaluate_model, save_results

IN_DOMAIN = ("Kvasir", "CVC-ClinicDB")
BINS = ("small", "medium", "large")
DARK = 0.05
KNN = 10
N_BOOT = 2000
BETA = 0.01
SEVERITIES = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------- image transforms (picklable classes)
def _valid(img: np.ndarray) -> np.ndarray:
    return img.max(axis=2) > DARK


def _lab_stats(img: np.ndarray):
    valid = _valid(img)
    if valid.sum() < 16:
        return None
    px = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)[valid]
    return px.mean(0), px.std(0)


class Reinhard:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        valid = _valid(img)
        if valid.sum() < 16:
            return img
        lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
        px = lab[valid]
        lab[valid] = (px - px.mean(0)) / (px.std(0) + 1e-6) * self.std + self.mean
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        out[~valid] = img[~valid]
        return out


class HistMatch:
    def __init__(self, cdf: np.ndarray):
        self.cdf = cdf                                                   # [3, 256] pooled training CDF

    def __call__(self, img: np.ndarray) -> np.ndarray:
        valid = _valid(img)
        if valid.sum() < 16:
            return img
        q = np.clip(np.round(img * 255), 0, 255).astype(np.int64)
        out = img.copy()
        for c in range(3):
            v = q[..., c][valid]
            cdf = np.cumsum(np.bincount(v, minlength=256)) / v.size
            lut = np.clip(np.searchsorted(self.cdf[c], cdf, side="left"), 0, 255) / 255.0
            channel = out[..., c]
            channel[valid] = lut[v]
        return out


class LowFreqAmplitude:
    def __init__(self, amp: np.ndarray, beta: float = BETA):
        self.amp, self.beta = amp, beta                                  # [3, S, S] unshifted mean amplitude

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.amp.shape[1:] != (h, w):
            raise ValueError("amplitude statistics were computed at a different image size")
        b = int(np.floor(min(h, w) * self.beta))
        f = np.fft.fft2(img.transpose(2, 0, 1).astype(np.float64))
        amp, pha = np.fft.fftshift(np.abs(f), axes=(-2, -1)), np.angle(f)
        src = np.fft.fftshift(self.amp, axes=(-2, -1))
        ch, cw = h // 2, w // 2
        amp[:, ch - b:ch + b + 1, cw - b:cw + b + 1] = src[:, ch - b:ch + b + 1, cw - b:cw + b + 1]
        out = np.real(np.fft.ifft2(np.fft.ifftshift(amp, axes=(-2, -1)) * np.exp(1j * pha)))
        return np.clip(out.transpose(1, 2, 0), 0, 1).astype(np.float32)


class Gamma:
    def __init__(self, severity: int):
        self.g = 1.0 + 0.3 * severity                                    # 1.3 ... 2.5 (darker)

    def __call__(self, img):
        return np.power(np.clip(img, 0, 1), self.g)


class ColourTemperature:
    def __init__(self, severity: int):
        self.k = 0.08 * severity                                         # red up, blue down (warmer)

    def __call__(self, img):
        out = img.copy()
        out[..., 0] *= 1 + self.k
        out[..., 2] *= 1 - self.k
        return np.clip(out, 0, 1)


class Compose:
    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, img):
        for fn in self.fns:
            img = fn(img)
        return img


# ---------------------------------------------------------------- statistics
def source_statistics(paths: list[str], size: int) -> dict:
    """One pass over the training IMAGES (never masks)."""
    desc, lab_m, lab_s, amp = [], [], [], np.zeros((3, size, size))
    hist = np.zeros((3, 256))
    for k, p in enumerate(paths):
        img = load_image(p)
        desc.append(descriptor(img))
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        st = _lab_stats(small)
        if st is not None:
            lab_m.append(st[0]); lab_s.append(st[1])
        valid = _valid(small)
        q = np.clip(np.round(small * 255), 0, 255).astype(np.int64)
        for c in range(3):
            hist[c] += np.bincount(q[..., c][valid], minlength=256)
        amp += np.abs(np.fft.fft2(small.transpose(2, 0, 1).astype(np.float64)))
        if (k + 1) % 250 == 0:
            print(f"    {k + 1}/{len(paths)} training images")
    return {"desc": np.stack(desc), "lab_mean": np.mean(lab_m, 0), "lab_std": np.mean(lab_s, 0),
            "cdf": np.cumsum(hist, axis=1) / hist.sum(axis=1, keepdims=True), "amp": amp / len(paths)}


def knn_distance(train_z: np.ndarray, query_z: np.ndarray, k: int = KNN) -> np.ndarray:
    out = []
    for i in range(0, len(query_z), 256):
        d = np.linalg.norm(query_z[i:i + 256, None] - train_z[None], axis=2)
        out.append(np.sort(d, axis=1)[:, :k].mean(1))
    return np.concatenate(out)


def partial_spearman(x, y, z):
    """Rank correlation of x and y with z held fixed (Pearson on rank residuals). Returns (rho, p)."""
    from scipy import stats
    if min(np.ptp(x), np.ptp(y), np.ptp(z)) == 0:                     # a constant factor: undefined
        return float("nan"), float("nan")
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    r = float(np.corrcoef(res(rx), res(ry))[0, 1])
    n = len(x)
    t = r * np.sqrt(max(n - 3, 1) / max(1 - r * r, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), max(n - 3, 1)))


def correlations(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    from scipy import stats
    rows = []
    for gname, members in groups.items():
        g = df[df["dataset"].isin(members)]
        if len(g) < 10:
            continue
        d, dist, area = g["dice"].to_numpy(), g["appearance_dist"].to_numpy(), np.log(g["area_pct"].to_numpy() + 1e-3)
        r_d, p_d = stats.spearmanr(d, dist)
        r_a, p_a = stats.spearmanr(d, area)
        pr_d, pp_d = partial_spearman(d, dist, area)
        pr_a, pp_a = partial_spearman(d, area, dist)
        rows.append({"group": gname, "n": len(g), "rho_dice_distance": r_d, "p_distance": p_d,
                     "rho_dice_area": r_a, "p_area": p_a, "partial_rho_distance_given_area": pr_d,
                     "p_partial_distance": pp_d, "partial_rho_area_given_distance": pr_a, "p_partial_area": pp_a})
    return pd.DataFrame(rows)


def oaxaca(d_in, b_in, d_un, b_un) -> tuple[float, float, float]:
    """Returns (gap, size-composition part, within-size part); the two parts sum exactly to the gap."""
    comp = within = 0.0
    for b in BINS:
        w_in, w_un = np.mean(b_in == b), np.mean(b_un == b)
        if w_in == 0 and w_un == 0:
            continue
        if w_in == 0:                                                    # bin absent in-domain: undefined
            return float("nan"), float("nan"), float("nan")
        D_in = d_in[b_in == b].mean()
        comp += (w_in - w_un) * D_in
        if w_un > 0:
            within += w_un * (D_in - d_un[b_un == b].mean())
    return float(d_in.mean() - d_un.mean()), float(comp), float(within)


def decomposition(df: pd.DataFrame, unseen: dict[str, list[str]], n_boot: int = N_BOOT, seed: int = 0):
    rng = np.random.default_rng(seed)
    ref = df[df["dataset"].isin(IN_DOMAIN)]
    d_in, b_in = ref["dice"].to_numpy(), ref["size_bin"].to_numpy()
    rows = []
    for gname, members in unseen.items():
        g = df[df["dataset"].isin(members)]
        if g.empty:
            continue
        d_un, b_un = g["dice"].to_numpy(), g["size_bin"].to_numpy()
        gap, comp, within = oaxaca(d_in, b_in, d_un, b_un)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(d_in), len(d_in))
            j = rng.integers(0, len(d_un), len(d_un))
            boots.append(oaxaca(d_in[i], b_in[i], d_un[j], b_un[j]))
        boots = np.array(boots)
        boots = boots[~np.isnan(boots).any(axis=1)]
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        rows.append({"unseen": gname, "n": len(g), "dice_in": d_in.mean(), "dice_unseen": d_un.mean(),
                     "gap": gap, "gap_lo": lo[0], "gap_hi": hi[0],
                     "size_part": comp, "size_lo": lo[1], "size_hi": hi[1],
                     "within_part": within, "within_lo": lo[2], "within_hi": hi[2],
                     "size_share": comp / gap if gap > 0 else float("nan"),
                     "small_share_in": np.mean(b_in == "small"), "small_share_unseen": np.mean(b_un == "small"),
                     "valid_boot": len(boots)})
    return pd.DataFrame(rows)


def miss_analysis(df: pd.DataFrame, unseen: dict[str, list[str]], miss: float = 0.1) -> pd.DataFrame:
    """Splits each gap into MISSED polyps (Dice < 0.1) and boundary quality of found polyps.
    extra_miss_part = (miss rate unseen - miss rate in-domain) x in-domain Dice of found polyps
                    = Dice the unseen set would regain if it missed no more often than in-domain."""
    def stats(g):
        found = g["dice"] >= miss
        small = g["size_bin"] == "small"
        return {"n": len(g), "dice": g["dice"].mean(), "miss_rate": 1 - found.mean(),
                "miss_rate_small": (1 - found[small].mean()) if small.any() else float("nan"),
                "miss_rate_not_small": (1 - found[~small].mean()) if (~small).any() else float("nan"),
                "dice_detected": g.loc[found, "dice"].mean()}
    ref = stats(df[df["dataset"].isin(IN_DOMAIN)])
    rows = [{"group": "in-domain", **ref}]
    for gname, members in unseen.items():
        g = df[df["dataset"].isin(members)]
        if g.empty:
            continue
        r = stats(g)
        gap = ref["dice"] - r["dice"]
        part = (r["miss_rate"] - ref["miss_rate"]) * ref["dice_detected"]
        rows.append({"group": gname, **r, "gap": gap, "extra_miss_part": part,
                     "extra_miss_share": part / gap if gap > 0 else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def run_diagnosis(source_run_dir: str | Path, out_dir: str | Path, loc, device=None,
                  include_external: bool = True, n_boot: int = N_BOOT) -> Path:
    out_dir = Path(out_dir)
    passes = out_dir / "passes"
    passes.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = build_trained_model(source_run_dir, device)
    size, amp, nw = cfg["data"]["img_size"], cfg["train"]["amp"], cfg["data"]["num_workers"]
    print(f"[C] diagnosing {source_run_dir} ({cfg['name']}, seed {cfg['seed']}, input {size} px)")

    sets = test_sets(loc.data, include_external)
    in_sets = {k: v for k, v in sets.items() if k in IN_DOMAIN}
    un_sets = {k: v for k, v in sets.items() if k not in IN_DOMAIN}

    def evaluate(tag: str, subsets: dict, transform=None, full=False) -> pd.DataFrame:
        path = passes / f"{tag}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"id": str})        # ids such as "12" must stay strings
        print(f"  pass: {tag}")
        df = evaluate_model(model, subsets, size, device, num_workers=nw, amp=amp, transform=transform,
                            verbose=False, full_metrics=full)
        df["id"] = df["id"].astype(str)
        df.to_csv(path, index=False)
        return df

    # 1. source statistics and per-image factors
    stats_path = out_dir / "source_stats.npz"
    if stats_path.exists():
        st = dict(np.load(stats_path))
    else:
        print("[1/5] statistics of the 1,450 training images (images only)")
        st = source_statistics([s["image"] for s in pranet_train(loc.data)], size)
        np.savez_compressed(stats_path, **st)
    scaler = Standardiser(st["desc"])
    train_z = scaler(st["desc"])

    print("[2/5] baseline evaluation and appearance distance")
    base = evaluate("baseline", sets, full=True)
    save_results(base, out_dir / "baseline")
    paths = {(name, s["id"]): s["image"] for name, ss in sets.items() for s in ss}
    if "appearance_dist" not in base.columns:
        q = np.stack([descriptor(load_image(paths[(r.dataset, r.id)])) for r in base.itertuples()])
        base["appearance_dist"] = knn_distance(train_z, scaler(q))
        base.to_csv(passes / "baseline.csv", index=False)
    base.to_csv(out_dir / "per_image.csv", index=False)

    unseen_groups = {k: [k] for k in un_sets}
    polypgen = [k for k in un_sets if k.startswith("PolypGen")]
    if polypgen:
        unseen_groups["PolypGen-all"] = polypgen
    unseen_groups["unseen-all (excl. CVC-300)"] = [k for k in un_sets if k != "CVC-300"]
    corr = correlations(base, {**{k: [k] for k in sets}, "in-domain": list(IN_DOMAIN), **unseen_groups})
    corr.to_csv(out_dir / "correlations.csv", index=False)
    dec = decomposition(base, unseen_groups, n_boot)
    dec.to_csv(out_dir / "decomposition.csv", index=False)
    misses = miss_analysis(base, unseen_groups)
    misses.to_csv(out_dir / "misses.csv", index=False)

    # 3. normalisers on unseen sets
    print("[3/5] appearance normalisers on unseen sets")
    norms = {"reinhard": Reinhard(st["lab_mean"], st["lab_std"]), "histmatch": HistMatch(st["cdf"]),
             "lowfreq": LowFreqAmplitude(st["amp"])}
    d_in = base[base["dataset"].isin(IN_DOMAIN)]["dice"].mean()
    rows = []
    for nname, fn in norms.items():
        df = evaluate(f"unseen_{nname}", un_sets, fn)
        merged = base[["dataset", "id", "dice", "size_bin"]].merge(df[["dataset", "id", "dice"]],
                                                                    on=["dataset", "id"], suffixes=("", "_norm"))
        for gname, members in unseen_groups.items():
            g = merged[merged["dataset"].isin(members)]
            for b in ("all",) + BINS:
                gb = g if b == "all" else g[g["size_bin"] == b]
                if gb.empty:
                    continue
                gap = d_in - g["dice"].mean()
                rows.append({"normaliser": nname, "unseen": gname, "size_bin": b, "n": len(gb),
                             "dice": gb["dice"].mean(), "dice_normalised": gb["dice_norm"].mean(),
                             "recovered": gb["dice_norm"].mean() - gb["dice"].mean(),
                             "share_of_gap_recovered": (gb["dice_norm"].mean() - gb["dice"].mean()) / gap
                             if (b == "all" and gap > 0) else float("nan")})
    normal = pd.DataFrame(rows)
    normal.to_csv(out_dir / "normalisers.csv", index=False)

    # 4. controls
    print("[4/5] controls")
    crow = []
    base_in = base[base["dataset"].isin(IN_DOMAIN)].set_index(["dataset", "id"])["dice"]
    for nname, fn in norms.items():
        df = evaluate(f"negative_{nname}", in_sets, fn).set_index(["dataset", "id"])["dice"]
        crow.append({"control": "negative (in-domain normalised)", "normaliser": nname, "dice_before": base_in.mean(),
                     "dice_shifted": float("nan"), "dice_after": df.mean(), "change": df.mean() - base_in.mean()})
    if "ETIS-LaribPolypDB" in sets:
        etis = [_lab_stats(cv2.resize(load_image(s["image"]), (size, size))) for s in sets["ETIS-LaribPolypDB"]]
        etis = [e for e in etis if e is not None]
        shift = Reinhard(np.mean([e[0] for e in etis], 0), np.mean([e[1] for e in etis], 0))
        shifted = evaluate("positive_shift_to_etis", in_sets, shift)["dice"].mean()
        for nname, fn in norms.items():
            df = evaluate(f"positive_{nname}", in_sets, Compose(shift, fn))
            crow.append({"control": "positive (in-domain shifted to ETIS colour, then normalised)",
                         "normaliser": nname, "dice_before": base_in.mean(), "dice_shifted": shifted,
                         "dice_after": df["dice"].mean(), "change": df["dice"].mean() - shifted})
    controls = pd.DataFrame(crow)
    controls.to_csv(out_dir / "controls.csv", index=False)

    # 5. appearance sensitivity
    print("[5/5] appearance sensitivity (gamma, colour temperature)")
    arow = []
    for cname, cls in (("gamma", Gamma), ("colour_temperature", ColourTemperature)):
        dices = [evaluate(f"asi_{cname}_{s}", in_sets, cls(s))["dice"].mean() for s in SEVERITIES]
        for s, d in zip(SEVERITIES, dices):
            arow.append({"corruption": cname, "severity": s, "dice": d, "dice_clean": base_in.mean()})
        arow.append({"corruption": cname, "severity": "nAUC", "dice": np.mean(dices) / base_in.mean(),
                     "dice_clean": base_in.mean()})
        arow.append({"corruption": cname, "severity": "ASI", "dice": 1 - np.mean(dices) / base_in.mean(),
                     "dice_clean": base_in.mean()})
    asi = pd.DataFrame(arow)
    asi.to_csv(out_dir / "asi.csv", index=False)

    _figures(base, dec, out_dir)
    _report(cfg, source_run_dir, base, corr, dec, misses, normal, controls, asi, out_dir)
    print(f"[done] diagnosis written to {out_dir}")
    return out_dir


def _figures(base: pd.DataFrame, dec: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in base.groupby("dataset"):
        axes[0].scatter(g["appearance_dist"], g["dice"], s=6, alpha=0.5, label=name)
        axes[1].scatter(g["area_pct"], g["dice"], s=6, alpha=0.5, label=name)
    axes[0].set_xlabel("appearance distance to training images (10-NN, standardised)")
    axes[1].set_xlabel("polyp area (% of image, log scale)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("Dice")
    axes[1].legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "dice_vs_factors.png", dpi=150)
    plt.close(fig)
    if not dec.empty:
        fig, ax = plt.subplots(figsize=(1.2 * len(dec) + 3, 4))
        x = np.arange(len(dec))
        ax.bar(x, dec["size_part"], label="size composition", color="#4c72b0")
        ax.bar(x, dec["within_part"], bottom=dec["size_part"].clip(lower=0), label="within size (appearance, other)",
               color="#dd8452")
        ax.errorbar(x, dec["gap"], yerr=[dec["gap"] - dec["gap_lo"], dec["gap_hi"] - dec["gap"]], fmt="k.",
                    capsize=3, label="total gap (95 % CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(dec["unseen"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Dice gap to in-domain")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "gap_decomposition.png", dpi=150)
        plt.close(fig)


def _report(cfg, source, base, corr, dec, misses, normal, controls, asi, out_dir: Path) -> None:
    f = lambda v: "nan" if pd.isna(v) else f"{v:.3f}"
    L = [f"Diagnosis of {cfg['name']} seed {cfg['seed']} ({source})",
         "All numbers measured; interpretation is left to the reader / NOVELTY.md.", "",
         "Dice by dataset and polyp size (threshold 0.5, native resolution):"]
    tab = base.pivot_table(index="dataset", columns="size_bin", values="dice", aggfunc="mean")
    cnt = base.groupby(["dataset", "size_bin"]).size().unstack(fill_value=0)
    for ds in tab.index:
        L.append(f"  {ds:24s} all {f(base[base['dataset'] == ds]['dice'].mean())}  " + "  ".join(
            f"{b} {f(tab.loc[ds].get(b))} (n={cnt.loc[ds].get(b, 0)})" for b in BINS))
    L += ["", "Gap decomposition vs in-domain (Kvasir + ClinicDB test) [95 % bootstrap CI]:"]
    for r in dec.itertuples():
        L.append(f"  {r.unseen:28s} gap {f(r.gap)} [{f(r.gap_lo)}, {f(r.gap_hi)}] = size part {f(r.size_part)} "
                 f"[{f(r.size_lo)}, {f(r.size_hi)}] + within-size {f(r.within_part)} [{f(r.within_lo)}, "
                 f"{f(r.within_hi)}]; small polyps {r.small_share_in:.0%} in-domain vs {r.small_share_unseen:.0%}")
    L += ["", "Missed polyps (Dice < 0.1) vs boundary quality of found polyps:"]
    for r in misses.itertuples():
        extra = "" if r.group == "in-domain" else (f"  gap {f(r.gap)}, of which extra misses {f(r.extra_miss_part)} "
                                                   f"({f(r.extra_miss_share)} of the gap)")
        L.append(f"  {r.group:28s} miss {r.miss_rate:.1%} (small {r.miss_rate_small:.1%}, others "
                 f"{r.miss_rate_not_small:.1%})  Dice of found polyps {f(r.dice_detected)}" + extra)
    L += ["", "Correlations (Spearman; partial = other factor held fixed):"]
    for r in corr.itertuples():
        L.append(f"  {r.group:28s} n={r.n:5d}  dice~distance {f(r.rho_dice_distance)} (p={r.p_distance:.1e})  "
                 f"dice~area {f(r.rho_dice_area)} (p={r.p_area:.1e})  partial distance {f(r.partial_rho_distance_given_area)}"
                 f"  partial area {f(r.partial_rho_area_given_distance)}")
    L += ["", "Appearance normalisers on unseen sets (Dice recovered = lower bound on appearance-caused loss):"]
    for r in normal[normal["size_bin"] == "all"].itertuples():
        L.append(f"  {r.normaliser:10s} {r.unseen:28s} {f(r.dice)} -> {f(r.dice_normalised)}  recovered "
                 f"{f(r.recovered)}  share of gap {f(r.share_of_gap_recovered)}")
    L += ["", "Controls:"]
    for r in controls.itertuples():
        L.append(f"  {r.control:62s} {r.normaliser:10s} before {f(r.dice_before)} shifted {f(r.dice_shifted)} "
                 f"after {f(r.dice_after)} change {f(r.change)}")
    L += ["", "Appearance sensitivity on in-domain images:"]
    for r in asi.itertuples():
        L.append(f"  {r.corruption:20s} {str(r.severity):5s} {f(r.dice)}")
    (out_dir / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
