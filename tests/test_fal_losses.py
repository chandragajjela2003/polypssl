"""FAL and loss: the faithful versions reproduce the released code's behaviour (including its issues),
the corrected versions do what the paper describes."""
import torch

from polypssl.fal import _amplitude_mix, fal
from polypssl.losses import ari_loss, deep_supervision_loss, logit_margin_l1_faithful


def test_faithful_fal_is_a_per_channel_constant_shift():
    """Released FAL at 320 px blends only the DC term -> output = input + constant per channel."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 3, 320, 320, generator=g)
    copy = _amplitude_mix(x, x.roll(1, 0), half_width=int(320 * 0.001), lam=0.5)
    residual = copy - x
    per_channel_spread = (residual - residual.mean(dim=(-2, -1), keepdim=True)).abs().max()
    assert int(320 * 0.001) == 0
    assert per_channel_spread < 1e-4


def test_corrected_fal_changes_only_the_low_frequency_amplitude():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, 64, 64, generator=g)
    b = 3
    y = _amplitude_mix(x, x.flip(0), half_width=b, lam=0.5)
    fx = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    fy = torch.fft.fftshift(torch.fft.fft2(y), dim=(-2, -1))
    outside = torch.ones(64, 64, dtype=torch.bool)
    outside[32 - b:32 + b + 1, 32 - b:32 + b + 1] = False
    assert torch.allclose(fx.abs()[..., outside], fy.abs()[..., outside], atol=1e-3)
    big = fx.abs() > 1e-2                                 # phase is only defined where amplitude is non-zero
    dphase = torch.angle(fy * fx.conj())
    assert dphase[big].abs().max() < 1e-3                 # phase kept everywhere
    assert not torch.allclose(fx.abs()[..., ~outside], fy.abs()[..., ~outside])


def test_fal_stripes_and_range():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(5, 3, 60, 60, generator=g)
    out = fal(x, "corrected", torch.Generator().manual_seed(3), beta=0.05)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    # the first third (weight 0) is the original image, untouched
    for i in range(5):
        top_ok = torch.allclose(out[i, :, :20, :], x[i, :, :20, :], atol=1e-5)
        left_ok = torch.allclose(out[i, :, :, :20], x[i, :, :, :20], atol=1e-5)
        assert top_ok or left_ok


def test_released_logit_margin_l1_is_always_zero():
    prob = torch.rand(3, 1, 32, 32)
    assert logit_margin_l1_faithful(prob).item() == 0.0


def test_corrected_loss_is_pixel_weighted_but_faithful_is_not():
    """Same mean error, placed at the boundary vs in the interior: only the corrected BCE notices."""
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1
    boundary = torch.zeros_like(mask)
    boundary[..., 16:18, 16:48] = 1                       # error band on the polyp boundary
    interior = torch.zeros_like(mask)
    interior[..., 30:32, 16:48] = 1                       # same number of pixels, polyp interior
    def prob_with_error(err):
        return (mask * (1 - 0.6 * err) + (1 - mask) * 0.0).clamp(1e-4, 1 - 1e-4)
    f_b, f_i = (ari_loss(prob_with_error(e), mask, "faithful", return_parts=True) for e in (boundary, interior))
    c_b, c_i = (ari_loss(prob_with_error(e), mask, "corrected", return_parts=True) for e in (boundary, interior))
    for part in ("bce", "mae"):
        assert torch.isclose(f_b[part], f_i[part], atol=1e-6)   # released code: placement ignored
        assert c_b[part] > c_i[part] + 1e-4                     # corrected: boundary errors weigh more


def test_deep_supervision_handles_five_outputs_and_single_output():
    mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
    five = [torch.randn(2, 1, 64 // 2 ** k, 64 // 2 ** k) for k in range(5)]
    assert torch.isfinite(deep_supervision_loss(five, mask, "corrected"))
    assert torch.isfinite(deep_supervision_loss(five, mask, "faithful"))
    assert torch.isfinite(deep_supervision_loss([five[0]], mask, "corrected"))
