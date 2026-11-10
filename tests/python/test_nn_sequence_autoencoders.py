"""Native recurrent autoencoder graph, validation and owning handles."""
import gc
import math

import pytest
from pydatamunge import nn


def test_native_sequence_bottleneck_owns_states_and_keeps_all_gradients():
    p = nn.ParameterStore(1871)
    images = nn.from_vector([((i * 13) % 31) / 30 for i in range(1568)], [2, 28, 28, 1])
    sequence = nn.ImageRowSequence(images)
    assert tuple(sequence.shape()) == (2, 28, 28)
    assert tuple(sequence.values()) == tuple(images.values())
    result = nn.RecurrentAutoencoder(p, sequence)
    assert p.size() == 8 and p.count() == 4956
    assert tuple(result.code.shape()) == (2, 32)
    assert tuple(result.encoder_states.shape()) == (2, 28, 32)
    assert tuple(result.decoder_states.shape()) == (2, 28, 32)
    code, reconstruction = result.code, result.reconstruction
    expected = tuple(reconstruction.values())
    del result
    gc.collect()
    assert tuple(reconstruction.values()) == expected
    assert tuple(nn.RecurrentDecode(p, code).reconstruction.values()) == expected
    gradients = nn.grad(nn.mse(reconstruction, sequence), p.parameters())
    assert len(gradients) == 8
    assert all(all(math.isfinite(v) for v in g.values()) and any(v != 0 for v in g.values())
               for g in gradients)
    assert not nn.RecurrentEncode(p, sequence).reconstruction.defined()
    assert not nn.RecurrentDecode(p, code).encoder_states.defined()


def test_native_sequence_input_gradients_and_invalid_geometry():
    x = nn.from_vector_double([.1, .6, .7, .2, .4, .8], [1, 3, 2]).requires_grad_()
    p = nn.ParameterStore(17, nn.DType.Float64)
    encoded = nn.RecurrentEncode(p, x, 3, 2, 2)
    derivative = nn.grad(nn.sum(encoded.code), [x])[0]
    assert all(any(v != 0 for v in nn.select(derivative, 1, t).values()) for t in range(3))
    assert tuple(nn.grad(nn.sum(nn.ImageRowSequence(x, 3, 2)), [x])[0].values()) == (1.,) * 6
    empty = nn.ParameterStore(1)
    for bad in (nn.zeros([1, 3, 2], nn.DType.Int32), nn.zeros([1, 6]), nn.full([1, 3, 2], math.nan)):
        with pytest.raises(RuntimeError):
            nn.RecurrentAutoencoder(empty, bad, 3, 2, 2)
    with pytest.raises(RuntimeError):
        nn.RecurrentAutoencoder(empty, nn.zeros([1, 3, 2]), 3, 2, 6)
    assert empty.size() == 0
