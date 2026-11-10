"""Native masked-autoencoder arithmetic, selectors and owning binding handles."""
import gc
import math

import pytest
from pydatamunge import nn


def test_patch_inverse_gradients_and_discrete_selector_contract():
    image = nn.from_vector([float(i) for i in range(784)], [1, 28, 28, 1]).requires_grad_()
    patches = nn.MaePatches(image)
    assert list(nn.MaeImage(patches).values()) == list(image.values())
    assert nn.MaePixel(5, 0) == 203
    assert list(nn.grad(nn.sum(nn.MaeImage(patches)), [image])[0].values()) == [1.] * 784
    ids = nn.cast(nn.from_vector_double([0., 4294967295.], [2]), nn.DType.Int64)
    for seed in (0, 4294967295):
        selector = nn.MaeSelection(ids, 0, seed)
        assert tuple(selector.shape()) == (2, 4, 16)
        assert not selector.requires_grad()
        assert list(selector.values()) == list(nn.MaeSelection(ids, 0, seed).values())
        assert list(nn.sum(selector, [1]).values()).count(1.) == 8
    assert list(nn.MaePosition(4).values())[:4] == [0., 1., 0., 1.]
    with pytest.raises(RuntimeError):
        nn.MaeSelection(nn.from_vector([1.], [1]))
    with pytest.raises(RuntimeError):
        nn.MaePixel(16, 0)


def test_native_model_owned_endpoints_target_free_decode_and_gradients():
    p = nn.ParameterStore(1877)
    image = nn.from_vector([((i * 17) % 97) / 96 for i in range(1568)], [2, 28, 28, 1])
    selector = nn.MaeSelection(nn.cast(nn.from_vector([7., 11.], [2]), nn.DType.Int64))
    result = nn.MaskedAutoencoder(p, image, selector)
    assert p.size() == 35 and p.count() == 13649
    assert tuple(result.code.shape()) == (2, 4, 32)
    assert tuple(result.encoder_attention.shape()) == (2, 2, 4, 4)
    assert tuple(result.decoder_attention.shape()) == (2, 2, 16, 16)
    assert sum(result.mask.values()) == 24
    code, prediction, target, mask = result.code, result.prediction, result.patches, result.mask
    expected = tuple(prediction.values())
    del result
    gc.collect()
    assert tuple(prediction.values()) == expected
    assert tuple(nn.MaeDecode(p, code, selector).prediction.values()) == expected
    derivatives = nn.grad(nn.MaeLoss(prediction, target, mask), p.parameters())
    assert len(derivatives) == 35
    assert all(all(math.isfinite(v) for v in g.values()) and any(v != 0 for v in g.values())
               for g in derivatives)
    encoded = nn.MaeEncode(p, image, selector)
    assert not encoded.prediction.defined()
    decoded = nn.MaeDecode(p, encoded.code, selector)
    assert not decoded.patches.defined()
    assert tuple(decoded.prediction.values()) == expected


def test_invalid_selector_fails_before_parameters_and_masked_loss_excludes_visible():
    p = nn.ParameterStore(1)
    image = nn.zeros([1, 28, 28, 1])
    selector = nn.MaeSelection(nn.cast(nn.from_vector([0.], [1]), nn.DType.Int64))
    for bad in (nn.zeros([1, 4, 16]), selector.clone().requires_grad_()):
        with pytest.raises(RuntimeError):
            nn.MaskedAutoencoder(p, image, bad)
        assert p.size() == 0
    prediction = nn.full([1, 2, 2], .3).requires_grad_()
    target = nn.full([1, 2, 2], .7)
    mask = nn.from_vector([0., 1.], [1, 2])
    loss = nn.MaeLoss(prediction, target, mask)
    assert loss.item() == pytest.approx(.16, abs=1e-7)
    assert list(nn.grad(loss, [prediction])[0].values()) == pytest.approx([0., 0., -.4, -.4])
