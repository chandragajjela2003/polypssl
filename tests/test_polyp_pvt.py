"""Polyp-PVT port: shapes, final prediction rule, UAL plug-in, official weights, structure loss, training recipe.
Set POLYPSSL_TEST_WEIGHTS to the folder containing pvt_v2_b2.pth to run the weight-loading test."""
import os
from pathlib import Path

import pytest
import torch

from polypssl.losses import segmentation_loss, structure_loss
from polypssl.models import final_logits
from polypssl.models.polyp_pvt import PolypPVT
from polypssl.models.pvtv2 import load_pvt_weights, pvt_v2_b2


def test_outputs_and_final_prediction():
    model = PolypPVT(ual="off").eval()
    with torch.no_grad():
        outs = model(torch.randn(2, 3, 352, 352))
    assert [tuple(o.shape) for o in outs] == [(2, 1, 352, 352)] * 2
    assert torch.allclose(final_logits(model, outs), outs[0] + outs[1])   # official Test.py: P1 + P2


def test_backbone_pyramid():
    feats = pvt_v2_b2().eval()(torch.randn(1, 3, 320, 320))
    assert [tuple(f.shape[1:]) for f in feats] == [(64, 80, 80), (128, 40, 40), (320, 20, 20), (512, 10, 10)]


def test_ual_plugin_changes_only_the_sam_branch():
    torch.manual_seed(0)
    off, on = PolypPVT(ual="off").eval(), PolypPVT(ual="corrected").eval()
    on.load_state_dict(off.state_dict())                  # UAL adds no parameters
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        a, b = off(x), on(x)
    assert torch.allclose(a[0], b[0])                     # P1 (CFM) is upstream of UAL: unchanged
    rel = (a[1] - b[1]).abs().mean() / a[1].abs().mean()
    assert rel > 1e-2                                     # P2 really changes (the T2 placement gave ~1e-6)


def test_faithful_structure_bce_ignores_boundary_weights():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1
    logits = torch.randn(1, 1, 64, 64)
    # faithful wbce = mean BCE * sum(weit)/sum(weit) = plain mean BCE (+ wIoU): weights have no effect on BCE
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    weit = 1 + 5 * torch.abs(torch.nn.functional.avg_pool2d(mask, 31, 1, 15) - mask)
    pred = torch.sigmoid(logits)
    inter, union = ((pred * mask) * weit).sum(), ((pred + mask) * weit).sum()
    wiou = 1 - (inter + 1) / (union - inter + 1)
    assert torch.isclose(structure_loss(logits, mask, "faithful"), plain + wiou, atol=1e-6)
    assert not torch.isclose(structure_loss(logits, mask, "corrected"), plain + wiou, atol=1e-4)


def test_structure_loss_sums_over_outputs():
    mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
    outs = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    cfg = {"type": "structure", "mode": "corrected"}
    total = segmentation_loss(outs, mask, cfg)
    assert torch.isclose(total, structure_loss(outs[0], mask, "corrected") + structure_loss(outs[1], mask, "corrected"))


WEIGHTS = os.environ.get("POLYPSSL_TEST_WEIGHTS")


@pytest.mark.skipif(not WEIGHTS or not (Path(WEIGHTS) / "pvt_v2_b2.pth").exists(),
                    reason="set POLYPSSL_TEST_WEIGHTS to the folder with pvt_v2_b2.pth")
def test_official_pvt_weights_load_completely():
    model = pvt_v2_b2()
    report = load_pvt_weights(model, Path(WEIGHTS) / "pvt_v2_b2.pth")
    assert report["missing"] == []
    assert all(k.startswith("head") for k in report["unused"])   # only the ImageNet classifier is unused
