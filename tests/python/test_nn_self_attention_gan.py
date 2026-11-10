"""Native spectral patch attention, owned records and host-owned state/D/state/G writes."""
import gc
import math
import pytest
from pydatamunge import nn


def test_shared_dense_and_unscaled_attention_zero_gate_owned_fields():
    p, s = nn.ParameterStore(4), nn.ParameterStore(0)
    x = nn.full([2, 3, 8], .2).requires_grad_()
    out = nn.SpectralSpatialAttention(p, s, x)
    assert list(out.input.values()) == list(out.output.values())
    assert tuple(out.probability.shape()) == (2, 3, 3)
    before = list(s.flatten())
    nn.SpectralSpatialAttention(p, s, x)
    assert list(s.flatten()) == before
    nn.advance_cached_spectral_state(p, s, 16)
    p.set('attention/gamma', nn.scalar(.3))
    out = nn.SpectralSpatialAttention(p, s, x)
    y, probability = out.output, out.probability
    del out, p, s
    gc.collect()
    assert all(math.isfinite(v) for v in probability.values())
    assert any(abs(v) > 1e-8 for v in nn.grad(nn.sum(y), [x])[0].values())
    d, state = nn.ParameterStore(1), nn.ParameterStore(0)
    assert tuple(nn.cached_spectral_dense(d, state, 'x', nn.ones([2, 3]), 4).shape()) == (2, 4)
    assert [d.count(), state.count()] == [16, 7]


def test_default_census_fields_gamma_identity_and_immutable_inference():
    stores = [nn.ParameterStore(seed) for seed in [5903, 5923, 0, 0]]
    x, z = nn.zeros([2, 784]), nn.zeros([3, 16])
    out = nn.SelfAttentionGan(*stores, x, z)
    assert [p.count() for p in stores] == [2730, 650, 245, 110]
    assert [p.size() for p in stores] == [13, 14, 12, 12]
    assert tuple(out.fake.shape()) == (3, 784) and tuple(out.real_score.shape()) == (2, 1)
    assert list(out.g_attention_input.values()) == list(out.g_attention_output.values())
    assert list(out.real_attention_input.values()) == list(out.real_attention_output.values())
    before = [list(p.flatten()) for p in stores]
    nn.SelfAttentionGan(*stores, x, z)
    assert before == [list(p.flatten()) for p in stores]


def test_four_explicit_phases_write_only_own_parameter_or_cache_store():
    stores = [nn.ParameterStore(seed) for seed in [5903, 5923, 0, 0]]
    g, d, gs, ds = stores
    x, z = nn.full([2, 784], .3), nn.full([3, 16], -.2)
    forward = lambda: nn.SelfAttentionGan(*stores, x, z)
    forward()
    for p, s in [(g, gs), (d, ds)]: nn.advance_cached_spectral_state(p, s, 16)
    config = nn.OptimizerConfig()
    config.kind, config.beta1 = nn.OptimizerKind.Adam, .5
    go, disc = nn.Optimizer(g, .0002, config), nn.Optimizer(d, .0002, config)
    actions = [(3, lambda: nn.advance_cached_spectral_state(d, ds)),
               (1, lambda: disc.step(nn.grad(forward().discriminator_loss, list(d.parameters())))),
               (2, lambda: nn.advance_cached_spectral_state(g, gs)),
               (0, lambda: go.step(nn.grad(forward().generator_loss, list(g.parameters()))))]
    for owner, action in actions:
        before = [list(p.flatten()) for p in stores]
        action()
        after = [list(p.flatten()) for p in stores]
        assert after[owner] != before[owner]
        assert all(before[i] == after[i] for i in range(4) if i != owner)
    assert go.iterations() == disc.iterations() == 1


def test_generate_discriminate_and_full_record_fields_outlive_parent_and_stores():
    stores = [nn.ParameterStore(seed, nn.DType.Float64) for seed in [5903, 5923, 0, 0]]
    z = nn.full([2, 16], .2, nn.DType.Float64).requires_grad_()
    image = nn.full([2, 784], .3, nn.DType.Float64).requires_grad_()
    g = nn.SelfAttentionGanGenerate(stores[0], stores[2], z)
    d = nn.SelfAttentionGanDiscriminate(stores[1], stores[3], image)
    nn.advance_cached_spectral_state(stores[0], stores[2], 16)
    nn.advance_cached_spectral_state(stores[1], stores[3], 16)
    full = nn.SelfAttentionGan(*stores, image, z)
    pixels, score, loss = g.pixels, d.score, full.generator_loss
    del g, d, full, stores
    gc.collect()
    assert tuple(pixels.shape()) == (2, 784) and tuple(score.shape()) == (2, 1)
    assert any(abs(v) > 1e-10 for v in nn.grad(loss, [z])[0].values())


def test_bad_inputs_stores_and_geometry_precede_leaf_creation():
    g, d, gs, ds = [nn.ParameterStore(seed) for seed in [1, 2, 0, 0]]
    x, z = nn.zeros([2, 784]), nn.zeros([3, 16])
    with pytest.raises(RuntimeError): nn.SelfAttentionGan(g, g, gs, ds, x, z)
    with pytest.raises(RuntimeError): nn.SelfAttentionGan(g, d, gs, gs, x, z)
    with pytest.raises(RuntimeError): nn.SelfAttentionGan(g, d, gs, ds, x, nn.zeros([3, 15]))
    with pytest.raises(RuntimeError): nn.SelfAttentionGan(g, d, gs, ds, nn.full([2, 784], math.nan), z)
    with pytest.raises(RuntimeError): nn.SpectralSpatialAttention(g, gs, nn.ones([2, 3, 8]), 0)
    with pytest.raises(RuntimeError): nn.cached_spectral_dense(g, gs, '', nn.ones([2, 3]), 4)
    assert sum(p.size() for p in [g, d, gs, ds]) == 0
