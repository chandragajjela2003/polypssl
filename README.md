# polypssl: label-efficient, cross-hospital polyp segmentation

M.Tech thesis code (Priya Gajjela): semi-supervised colorectal polyp segmentation with very few labelled
images (5% = 72 masks), evaluated on hospitals that are not part of the training data.
Base paper: FULNet (BSPC 2026). Reproduced baseline: Polyp-PVT.

## What is in this repository

| Path | Contents |
|---|---|
| `polypssl/models/` | FULNet-C (MSCAN-T) and Polyp-PVT (PVTv2-B2) ports, checked against the official code (identical outputs for identical weights and input) |
| `polypssl/fal.py`, `losses.py` | FULNet's frequency augmentation (FAL) and the loss functions, in "faithful" (as released) and "corrected" modes |
| `polypssl/engine/` | Supervised training and Mean Teacher semi-supervised training, with automatic resume after a disconnect |
| `polypssl/ssl_views.py` | Strong photometric view and ZoomMatch (zoomed-out mosaics that create small polyps) |
| `polypssl/selection.py`, `appearance.py` | ACLS label selection (appearance clusters) and its composition-matched control |
| `polypssl/analysis/diagnosis.py` | Diagnosis of the cross-hospital Dice drop: polyp size vs appearance, missed polyps, normalisers, controls |
| `polypssl/evaluation/` | Metrics at native resolution: Dice, IoU, precision, recall, HD95, size bins, missed polyps |
| `experiments.yaml` | Every experiment by ID (A1 ... E2) and its settings |
| `splits/` | The exact labelled / unlabelled image lists for every budget and seed |
| `notebooks/colab`, `notebooks/kaggle` | One notebook per platform: set `RUN_ID` and `SEED`, run all cells |
| `tests/` | 79 unit tests (`python -m pytest tests`) |

## Data and protocol

- Training: PraNet split, Kvasir-SEG 900 + CVC-ClinicDB 550 images. Testing: Kvasir 100, ClinicDB 62,
  CVC-ColonDB 380, CVC-300 60, ETIS 196, plus PolypGen centres C1, C2, C3, C5, C6 when available.
- Label budgets are total masks (5% = 72 = 60 train + 12 validation); unlabelled masks are never read.
- No tuning or checkpoint selection on test data: the last checkpoint is reported.
- Datasets and pretrained ImageNet weights are downloaded by `setup_env.py`; they are not stored here.

## How to run

```bash
pip install -r requirements.txt
python setup_env.py                     # data, PolypGen (if its zip is present), pretrained weights
python run.py --list                    # every experiment and its status
python run.py --run B2 --seed 0         # train (resumes automatically) and evaluate
python run.py --results                 # results table of all finished runs
python run.py --run all --reevaluate    # evaluate finished models again (e.g. after adding a test set)
```

Results go to `outputs/<ID>__<name>/seed<k>/` (`model_final.pth`, `log.csv`, `config.yaml`,
`per_image.csv`, `summary.csv`) and `outputs/ALL_RESULTS.csv`.

## Experiments

| ID | Experiment |
|---|---|
| A1 | Polyp-PVT reproduction, 100% labels |
| A2, A3, A4 | FULNet reproductions (optional); PVT-FU at 100% labels |
| B1, B2 | 5% labels: supervised / Mean Teacher |
| C1, C2 | Diagnosis of A1 / B2 |
| D1, D2, D3 | Mean Teacher with ACLS labels / composition-matched control / DINOv2 control |
| E1, E2 | Mean Teacher with ZoomMatch / fixed 2x2 mosaic control |

## Acknowledgements

Model code is adapted from the official FULNet (github.com/zhaiying0925/FULNet) and Polyp-PVT
(github.com/DengPingFan/Polyp-PVT) repositories, which publish no licence; this repository is private
and for research and thesis use only.
