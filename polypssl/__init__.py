"""polypssl: label-efficient cross-hospital polyp segmentation (thesis/paper code).

Package layout
--------------
paths      where data and outputs live on Kaggle / Colab / local machines
config     YAML experiment configs
data       dataset discovery, label splits, augmentations, torch datasets
models     FULNet-C (MSCAN-T), UNet-ResNet34
losses     FULNet loss (faithful and corrected variants)
fal        Fourier low-frequency augmentation (faithful and corrected variants)
engine     training loop (supervised in Stage A; SSL methods are added later)
evaluation per-image metrics and evaluation over all test sets
utils      seeding, checkpointing, logging
"""

__version__ = "0.1.0"
