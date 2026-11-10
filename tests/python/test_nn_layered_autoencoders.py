"""Native configurable sigmoid layers and standalone decoder ownership."""
import gc
import math

import pytest
from pydatamunge import nn


def test_native_layer_list_owns_code_and_reconstruction_after_container_collection():
    p = nn.ParameterStore(1543)
    widths = [4, 3, 2, 3, 4]
    x = nn.from_vector([.1, .2, .3, .4, .5, .6, .7, .8], [2, 4])
    layers = nn.SigmoidAutoencoder(p, x, widths)
    assert isinstance(layers, list) and len(layers) == 4
    assert p.count() == 48 and p.size() == 8
    assert [tuple(t.shape()) for t in layers] == [(2, 3), (2, 2), (2, 3), (2, 4)]
    code, prediction = layers[1], layers[-1]
    expected = tuple(prediction.values())
    del layers
    gc.collect()
    assert tuple(prediction.values()) == expected
    assert tuple(nn.SigmoidAutoencoderDecode(p, code, widths).values()) == expected
    gradients = nn.grad(nn.mse(prediction, x), p.parameters())
    assert len(gradients) == 8
    assert all(all(math.isfinite(v) for v in g.values()) and any(v != 0 for v in g.values())
               for g in gradients)
    standalone = nn.ParameterStore(1553)
    output = nn.SigmoidAutoencoderDecode(standalone, nn.zeros([2, 16]), [784, 64, 16, 64, 784])
    assert tuple(output.shape()) == (2, 784)
    assert standalone.count() == 52048 and standalone.size() == 4
    assert not standalone.contains('layer0/W') and standalone.contains('layer2/W')


def test_native_geometry_and_decoder_reject_invalid_inputs_before_leaf_creation():
    p = nn.ParameterStore(1)
    x = nn.zeros([1, 4])
    for widths in ([], [4, 2], [4, 4, 4], [4, 2, 5], [4, 0, 4], [4, 2, 3, 2, 4]):
        with pytest.raises(RuntimeError):
            nn.SigmoidAutoencoder(p, x, widths)
    with pytest.raises(TypeError):
        nn.SigmoidAutoencoder(p, x, [4, 2.5, 4])
    with pytest.raises(RuntimeError):
        nn.SigmoidAutoencoderDecode(p, nn.zeros([1, 3]), [4, 2, 4])
    with pytest.raises(RuntimeError):
        nn.SigmoidAutoencoder(p, nn.full([1, 4], math.nan), [4, 2, 4])
    assert p.size() == 0
