"""Native Gumbel banks, relaxed/discrete contracts, endpoint ownership and host Adam."""
import gc
import math
import pytest
from pydatamunge import nn


def test_relaxation_kl_broadcast_derivatives_and_temperature_independent_ids():
    logits = nn.from_vector_double([.1, -.4, .7], [1, 3]).requires_grad_()
    noise = nn.from_vector_double([.3, -.2, .6, -.1, .4, .2], [2, 3]).requires_grad_()
    out = nn.GumbelCategorical(logits, noise, .7)
    q, y, ids = out.probability, out.relaxed, out.indices
    kl = nn.CategoricalUniformKl(q, out.log_probability)
    assert tuple(q.shape()) == (1, 3) and tuple(y.shape()) == (2, 3)
    expected = [math.exp(v) for v in [.1, -.4, .7]]
    expected = [v/sum(expected) for v in expected]
    assert list(q.values()) == pytest.approx(expected, abs=1e-14)
    assert kl.item() == pytest.approx(sum(v*math.log(3*v) for v in expected), abs=1e-14)
    objective = nn.add(nn.mul(nn.sum(nn.square(y)), .5), nn.mul(nn.sum(kl), .3))
    del out
    gc.collect()
    gradients = nn.grad(objective, [logits, noise])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)
    cold, warm = nn.GumbelCategorical(logits, noise, .25), nn.GumbelCategorical(logits, noise, 2)
    assert list(ids.values()) == list(cold.indices.values()) == list(warm.indices.values())
    assert list(cold.relaxed.values()) != list(warm.relaxed.values())
    assert not ids.requires_grad() and ids.dtype() == nn.DType.Int64


def test_explicit_noise_banks_own_lifetime_and_preserve_full_uint32_seed():
    bank = nn.GumbelNoise(2, 3, 4294967295, 1, 3)
    uniform, gumbel, noise = bank.uniform, bank.gumbel, bank.noise
    del bank
    gc.collect()
    assert tuple(noise.shape()) == (3, 2, 1, 3)
    assert uniform.dtype() == nn.DType.Float64 and noise.dtype() == nn.DType.Float32
    assert not noise.requires_grad()
    assert list(noise.values()) == list(nn.GumbelNoise(2, 3, 4294967295, 1, 3).noise.values())
    for u, g in zip(uniform.values(), gumbel.values()):
        assert 0 < u < 1 and g == pytest.approx(-math.log(-math.log(u)), abs=1e-14)
    assert list(noise.values()) == list(nn.cast(gumbel, nn.DType.Float32).values())
    with pytest.raises(RuntimeError):
        nn.GumbelNoise(2**62, 4, 1)


def test_all_model_fields_own_results_both_code_only_decoders_and_host_adam():
    t = nn.DType.Float64
    p = nn.ParameterStore(2903, t)
    image = nn.full([2, 3], .3, t)
    noise = nn.cast(nn.select(nn.GumbelNoise(2, 1, 2939, 2, 2).gumbel, 0, 0), t)
    out = nn.GumbelCategoricalAutoencoder(p, image, noise, .7, 3, 2, 2, 2)
    assert p.size() == 8 and p.count() == 39
    names = ['logits', 'log_probability', 'probability', 'relaxed', 'indices', 'hard', 'code',
             'soft_prediction', 'hard_prediction', 'nll', 'hard_nll', 'kl', 'loss', 'hard_objective']
    fields = {name: getattr(out, name) for name in names}
    del out
    gc.collect()
    assert fields['indices'].dtype() == nn.DType.Int32 and fields['hard'].dtype() == t
    assert not fields['indices'].requires_grad() and not fields['hard'].requires_grad()
    assert tuple(fields['code'].shape()) == (2, 4)
    assert list(nn.GumbelDecodeSoft(p, fields['relaxed'], 3, 2, 2, 2).values()) == list(fields['soft_prediction'].values())
    assert list(nn.GumbelDecodeIds(p, fields['indices'], 3, 2, 2, 2).values()) == list(fields['hard_prediction'].values())
    assert list(nn.GumbelDecodeIds(p, nn.cast(fields['indices'], nn.DType.Int64), 3, 2, 2, 2).values()) == list(fields['hard_prediction'].values())
    gradients = nn.grad(fields['loss'], list(p.parameters()))
    assert len(gradients) == 8 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_default_census_and_invalid_calls_before_creation():
    p = nn.ParameterStore(2903)
    nn.GumbelCategoricalAutoencoder(p, nn.zeros([1, 784]), nn.zeros([1, 4, 8]))
    assert p.count() == 53104 and p.size() == 8
    decoder = nn.ParameterStore(1)
    nn.GumbelDecodeSoft(decoder, nn.full([2, 4, 8], .125))
    assert decoder.count() == 26928 and decoder.size() == 4
    empty = nn.ParameterStore(1)
    for temperature in [0, -1, float('nan'), float('inf'), 1e-300]:
        with pytest.raises(RuntimeError):
            nn.GumbelCategoricalAutoencoder(empty, nn.zeros([2, 3]), nn.zeros([2, 2, 2]), temperature, 3, 2, 2, 2)
    with pytest.raises(RuntimeError):
        nn.GumbelDecodeSoft(empty, nn.zeros([1, 2, 2]), 3, 2, 2, 2)
    for bad in [-1, 2]:
        with pytest.raises(RuntimeError):
            nn.GumbelDecodeIds(empty, nn.full([1, 2], bad, nn.DType.Int64), 3, 2, 2, 2)
    assert empty.size() == 0
