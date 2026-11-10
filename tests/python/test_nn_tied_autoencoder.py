"""True native weight sharing, result ownership and target-free decoding."""
import gc
import math
import pytest
from pydatamunge import nn


def test_live_transpose_and_owning_result_fields_after_collection_and_updates():
    p = nn.ParameterStore(1853)
    x = nn.from_vector([0,0,1,0,0,1], [2,3])
    out = nn.TiedAutoencoder(p, x, 3, 2)
    code, prediction, decoder = out.code, out.reconstruction, out.decoder_weight
    del out
    gc.collect()
    assert p.size() == 3 and p.count() == 11 and not p.contains('decoder/W')
    assert tuple(nn.TiedAutoencoderDecode(p, code, 3, 2).values()) == tuple(prediction.values())
    gradients = nn.grad(nn.mse(prediction, x), p.parameters())
    assert len(gradients) == 3
    assert all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    options = nn.OptimizerConfig()
    options.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p,.003,options)
    initial = tuple(p.at('encoder/W').values())
    optimizer.minimize(nn.mse(nn.TiedAutoencoder(p, x, 3, 2).reconstruction, x))
    weight = p.at('encoder/W')
    assert tuple(weight.values()) != initial
    expected = tuple(nn.transpose(weight, [1, 0]).contiguous().values())
    assert tuple(nn.TiedAutoencoder(p, x, 3, 2).decoder_weight.contiguous().values()) == expected
    assert tuple(decoder.contiguous().values()) == expected
    assert not p.contains('decoder/W')


def test_standalone_decoder_census_gradients_and_invalid_inputs_before_creation():
    for tied in (True, False):
        p = nn.ParameterStore(1853)
        code = nn.from_vector([-.2, .3, .5, 0], [2, 2]).requires_grad_()
        y = nn.TiedAutoencoderDecode(p, code, 3, 2, tied)
        assert p.size() == 2 and p.count() == 9
        assert not p.contains('encoder/b')
        assert p.contains('encoder/W') == tied
        gradients = nn.grad(nn.sum(y), [code, *p.parameters()])
        assert len(gradients) == 3
        assert all(any(v != 0 for v in g.values()) for g in gradients)
    p = nn.ParameterStore(1)
    for x, features, latent in [(nn.zeros([1, 3]), 3, 3),
                                (nn.zeros([0, 3]), 3, 2),
                                (nn.full([1, 3], math.nan), 3, 2)]:
        with pytest.raises(RuntimeError):
            nn.TiedAutoencoder(p, x, features, latent)
    assert p.size() == 0
    nn.TiedAutoencoder(p, nn.zeros([1, 3]), 3, 2, False)
    with pytest.raises(RuntimeError, match='decoder/W'):
        nn.TiedAutoencoder(p, nn.zeros([1, 3]), 3, 2)
