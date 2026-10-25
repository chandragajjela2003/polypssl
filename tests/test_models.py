"""Model shapes, UAL modes, and (when the file is available) that the released MSCAN-T weights load fully.

Set POLYPSSL_TEST_WEIGHTS to the folder containing mscan_t.pth to run the weight-loading test."""
import os
from pathlib import Path

import pytest
import torch

from polypssl.models.fulnet import FULNetC
from polypssl.models.mscan import build_mscan, load_mscan_weights


@pytest.mark.parametrize("ual", ["corrected", "faithful", "off"])
def test_fulnet_c_output_shapes(ual):
    model = FULNetC(ual=ual).eval()
    with torch.no_grad():
        outs = model(torch.randn(2, 3, 320, 320))
    assert [tuple(o.shape) for o in outs] == [(2, 1, 320 // 2 ** k, 320 // 2 ** k) for k in range(5)]


def test_mscan_feature_pyramid():
    feats = build_mscan("mscan_t").eval()(torch.randn(1, 3, 320, 320))
    assert [tuple(f.shape[1:]) for f in feats] == [(16, 160, 160), (32, 80, 80), (64, 40, 40),
                                                     (160, 20, 20), (256, 10, 10)]


def test_ual_modes_change_the_computation():
    torch.manual_seed(0)
    corrected = FULNetC(ual="corrected").eval()
    faithful = FULNetC(ual="faithful").eval()
    off = FULNetC(ual="off").eval()
    faithful.load_state_dict(corrected.state_dict())
    off.load_state_dict(corrected.state_dict())
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        a, b, c = corrected(x)[0], faithful(x)[0], off(x)[0]
    assert not torch.allclose(a, b) and not torch.allclose(a, c) and not torch.allclose(b, c)


WEIGHTS = os.environ.get("POLYPSSL_TEST_WEIGHTS")


@pytest.mark.skipif(not WEIGHTS or not (Path(WEIGHTS) / "mscan_t.pth").exists(),
                    reason="set POLYPSSL_TEST_WEIGHTS to the folder with mscan_t.pth")
def test_released_mscan_weights_load_without_missing_keys():
    model = build_mscan("mscan_t")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = load_mscan_weights(model, Path(WEIGHTS) / "mscan_t.pth")
    assert report["missing"] == []
    changed = sum(not torch.equal(before[k], v) for k, v in model.state_dict().items()
                  if not k.endswith("num_batches_tracked"))
    assert changed > 0.9 * len(before)          # the weights really replaced the random initialisation
    print("unexpected (not used by the encoder):", report["unexpected"][:10])


def test_unet_r34_shape():
    pytest.importorskip("segmentation_models_pytorch")
    from polypssl.models import UNetR34
    out = UNetR34(pretrained=False).eval()(torch.randn(1, 3, 320, 320))
    assert len(out) == 1 and tuple(out[0].shape) == (1, 1, 320, 320)
