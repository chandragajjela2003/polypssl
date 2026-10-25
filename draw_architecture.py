"""Draws the v7 system architecture to a PNG. Usage: python draw_architecture.py <output.png>"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE = "#1d2533", "#5b6577", "#b9c1cf"
EXIST, EXIST_E = "#ffffff", "#9aa3b2"
NEW, NEW_E = "#fbe4ec", "#b0245a"
ANA, ANA_E = "#dff2f3", "#0b7480"
BAND = "#f1f3f6"

fig, ax = plt.subplots(figsize=(19, 11.2), dpi=170)
ax.set_xlim(0, 190)
ax.set_ylim(0, 112)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, lines=(), kind="exist", status=None, tsize=10.5, lsize=8.6):
    fc, ec = {"exist": (EXIST, EXIST_E), "new": (NEW, NEW_E), "ana": (ANA, ANA_E)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6 if kind != "exist" else 1.2))
    ax.text(x + 1.6, y + h - 2.2, title, fontsize=tsize, weight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 1.6, y + h - 5.6 - i * 2.75, line, fontsize=lsize, color=MUTED, va="top")
    if status:
        done = status == "done"
        ax.text(x + w - 1.2, y + h - 2.2, "✓ in code" if done else "planned", fontsize=7.6, va="top", ha="right",
                color="#1f7a3a" if done else "#8a6d00", weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#e3f4e8" if done else "#fff4cc", ec="none"))


def arrow(x1, y1, x2, y2, label=None, dashed=False, lx=0, ly=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=INK,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=8, color=MUTED, ha="center")


# title + legend
ax.text(2, 109, "Label-efficient cross-hospital polyp segmentation: system architecture (plan v7)",
        fontsize=15, weight="bold", color=INK)
ax.text(2, 105.4, "Diagnose why few-label models fail on new hospitals (appearance vs polyp size), then treat each cause.",
        fontsize=10, color=MUTED)
for i, (fc, ec, lab) in enumerate([(EXIST, EXIST_E, "Existing method (reused / corrected)"),
                                   (NEW, NEW_E, "New contribution"), (ANA, ANA_E, "Analysis contribution")]):
    x0 = [104, 140, 163][i]
    ax.add_patch(FancyBboxPatch((x0, 105.2), 3.2, 2.2, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.4))
    ax.text(x0 + 4.2, 106.3, lab, fontsize=8.5, va="center", color=INK)

# bands
for x, w, t1, t2 in [(1, 47, "STAGE 1 · BEFORE TRAINING", "Which images get a doctor's mask?"),
                     (50, 87, "STAGE 2 · TRAINING (Mean Teacher)", "Learn from 60 masks + 1,378 unlabelled images"),
                     (139, 50, "STAGE 3 · TESTING + DIAGNOSIS", "Unseen hospitals, no retraining")]:
    ax.add_patch(FancyBboxPatch((x, 1), w, 100, boxstyle="round,pad=0,rounding_size=1.5", fc=BAND, ec="none"))
    ax.text(x + 1.5, 98.5, t1, fontsize=9, weight="bold", color=MUTED)
    ax.text(x + 1.5, 95.2, t2, fontsize=10.5, weight="bold", color=INK)

# ---- Stage 1
box(3, 79, 43, 12, "Training pool: 1,450 images", ["Kvasir-SEG 900 + CVC-ClinicDB 550", "(PraNet split; no masks used yet)"],
    status="done")
arrow(24.5, 79, 24.5, 74.6)
box(3, 47, 43, 27.5, "ACLS: label selection",
    ["(appearance-clustered label selection)", "1. appearance fingerprint per image", "   (central crop: low-freq amplitude + colour)",
     "2. k-means into 72 look-alike groups", "3. label the most typical image per group",
     "Baselines: random ✓, mix-matched random,", "DINOv2 selection (USL-style)"], kind="new", status="planned", lsize=8.3)
arrow(24.5, 47, 24.5, 42.6)
box(3, 27, 43, 15.5, "72 masks (5% budget)", ["60 for training + 12 run-check only", "(validation inside the budget,", "no tuning, last checkpoint)"],
    status="done")
box(3, 6, 43, 17, "1,378 unlabelled images", ["used without their masks", "Label splits: 2.5/5/10/20% x 3 seeds", "shipped with the code ✓"],
    status="done")
arrow(46, 34.5, 60, 62, label="", rad=-0.15)
ax.text(53.5, 44, "image +\nmask", fontsize=8, color=MUTED, ha="center")
arrow(46, 14, 60, 14, label="")
ax.text(53, 16, "unlabelled\nimage", fontsize=8, color=MUTED, ha="center")

# ---- Stage 2: student
box(60, 52, 48, 39, "Student network (main model)",
    ["Polyp-PVT: PVTv2-B2 encoder + CIM, CFM, SAM", "  (official code, output identical to original)",
     "+ FULNet plug-ins (corrected):", "  FAL: low-frequency Fourier style mixing",
     "  UAL: uncertainty re-weighting of the", "       CFM feature by the coarse prediction",
     "Second backbone: UNet-ResNet34 (vs RD-Net)", "FULNet-C (MSCAN-T): reproduction / audit"], status="done", lsize=8.3)
box(112, 66, 23.5, 21, "Training loss", ["masks: structure loss", "(pixel-weighted, bug fixed)",
                                         "+ pseudo-label loss", "  x confidence"], status=None)
arrow(108, 80, 112, 80, label="predict", ly=1.4)
arrow(112, 72, 108, 72, label="update", ly=-3.2)
# teacher
box(60, 6, 48, 18, "Teacher network", ["same architecture,", "slow EMA copy of the student weights", "(Mean Teacher)"], status="planned")
arrow(84, 52, 84, 24.4, label="EMA (slow copy)", lx=8, dashed=True)
# ZoomMatch
box(112, 6, 23.5, 54, "ZoomMatch", ["teaches the SMALL-polyp", "case missing from labels", "", "1. teacher predicts at", "   native scale -> area",
                                    "2. pick zoom so polyp", "   covers 0.3-15% of frame", "3. student sees a zoomed-", "   out MOSAIC of frames",
                                    "4. pseudo-label warped", "   identically", "", "(closest prior: MixPL", " Pseudo Mosaic, detection)"],
    kind="new", status="planned", lsize=8.1)
arrow(108, 15, 112, 15, label="", rad=0)
arrow(123.75, 60, 123.75, 66, label="")
ax.text(126, 62.3, "pseudo-label", fontsize=8, color=MUTED)
arrow(112, 46, 108, 56, label="", rad=0.1)
ax.text(103.5, 47, "zoomed-out\nview", fontsize=8, color=MUTED, ha="center")

# ---- Stage 3
box(141, 79, 46, 12.5, "Trained student", ["teacher discarded; last checkpoint only", "(never chosen on test data)"], status="done")
arrow(100, 91.5, 141, 89, label="", dashed=True, rad=-0.25)
ax.text(127, 97.2, "after training", fontsize=8, color=MUTED, ha="center")
arrow(164, 79, 164, 75.4)
box(141, 47.5, 46, 27.5, "Test sets (never trained on)",
    ["Block 2 · external hospitals (PRIMARY):", "  ETIS (Paris), PolypGen C1 C2 C3 C5 C6,", "  BKAI (Vietnam)",
     "Block 1 · FULNet comparison:", "  Kvasir, ClinicDB, ColonDB, ETIS,", "  CVC-300 (pixel-identical subset of ColonDB)"],
    status="done", lsize=8.3)
arrow(164, 47.5, 164, 43.6)
box(141, 30, 46, 13.5, "Metrics at original resolution", ["Dice, IoU, recall, precision, HD95 (%)",
                                                           "+ results by polyp size (<2%, 2-10%, >10%)"], status="done")
arrow(164, 30, 164, 26.6, dashed=True)
box(141, 4, 46, 22.5, "E1+ diagnosis: why does it fail?",
    ["share of the drop due to APPEARANCE", "  (recolour tests + controls)",
     "vs due to POLYP SIZE", "  (size-matched comparison)", "Gate 0: decides if ZoomMatch stays"],
    kind="ana", status="planned")

plt.savefig(sys.argv[1], bbox_inches="tight", facecolor="white")
print("saved", sys.argv[1])
