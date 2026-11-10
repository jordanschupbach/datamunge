"""Native cached spectral math, ownership, and host-owned state/D/G phases."""
import gc
import math
import pytest
from pydatamunge import nn


def test_cache_fields_live_denominator_stopped_pair_and_signed_floor():
    w = nn.from_vector_double([2], [1, 1]).requires_grad_()
    one = nn.ones([1, 1], nn.DType.Float64)
    result = nn.cached_spectral_kernel(w, one, one)
    effective, sigma = result.weight, result.sigma
    del result
    gc.collect()
    assert effective.item() == 1 and sigma.item() == 2
    assert nn.grad(nn.sum(effective), [w])[0].item() == 0
    negative = nn.cached_spectral_kernel(nn.from_vector_double([-2], [1, 1]), one, one, .5)
    assert negative.sigma.item() == -2 and negative.weight.item() == -4
    pair = nn.cached_spectral_power_iteration(nn.zeros([2, 3]), nn.ones([3, 1]))
    u, v = pair.u, pair.v
    del pair
    gc.collect()
    assert tuple(u.shape()) == (3, 1) and tuple(v.shape()) == (2, 1)
    assert not u.requires_grad() and not v.requires_grad()
    assert list(u.values()) == [0, 0, 0]
    for guard in [0, -1, math.nan, math.inf, 1e-300]:
        with pytest.raises(RuntimeError):
            nn.cached_spectral_power_iteration(nn.ones([2, 3]), nn.ones([3, 1]), 1, guard)


def test_default_census_and_explicit_state_then_D_then_fresh_G():
    g, d, state = [nn.ParameterStore(seed) for seed in [5801, 5807, 0]]
    x, z = nn.zeros([2, 784]), nn.zeros([3, 16])
    out = nn.SpectralGan(g, d, state, x, z)
    assert [p.count() for p in [g, d, state]] == [26416, 25153, 849]
    assert tuple(out.fake.shape()) == (3, 784) and tuple(out.real_score.shape()) == (2, 1)
    before = list(state.flatten())
    nn.SpectralGan(g, d, state, x, z)
    assert list(state.flatten()) == before
    nn.advance_cached_spectral_state(d, state, 16)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    config.beta1 = .5
    go, disc = nn.Optimizer(g, .0002, config), nn.Optimizer(d, .0002, config)
    gb, db = list(g.flatten()), list(d.flatten())
    nn.advance_cached_spectral_state(d, state)
    sb = list(state.flatten())
    assert list(g.flatten()) == gb and list(d.flatten()) == db
    first = nn.SpectralGan(g, d, state, x, z)
    disc.step(nn.grad(first.discriminator_loss, list(d.parameters())))
    assert list(g.flatten()) == gb and list(state.flatten()) == sb
    assert list(d.flatten()) != db
    db = list(d.flatten())
    fresh = nn.SpectralGan(g, d, state, x, z)
    go.step(nn.grad(fresh.generator_loss, list(g.parameters())))
    assert list(g.flatten()) != gb and list(d.flatten()) == db and list(state.flatten()) == sb
    assert go.iterations() == disc.iterations() == 1


def test_model_fields_retain_graph_after_parent_and_stores_are_collected():
    t = nn.DType.Float64
    g, d, state = [nn.ParameterStore(seed, t) for seed in [3, 4, 0]]
    x, z = nn.full([2, 3], .3, t), nn.full([3, 2], -.2, t).requires_grad_()
    nn.SpectralGan(g, d, state, x, z, 3, 2)
    nn.advance_cached_spectral_state(d, state, 16)
    out = nn.SpectralGan(g, d, state, x, z, 3, 2)
    fields = [getattr(out, key) for key in ['fake', 'real_score', 'fake_score', 'discriminator_loss', 'generator_loss']]
    del out, g, d, state
    gc.collect()
    assert all(all(math.isfinite(v) for v in field.values()) for field in fields)
    assert any(abs(v) > 1e-12 for v in nn.grad(fields[-1], [z])[0].values())


def test_invalid_stores_inputs_and_missing_cache_fail_without_writes():
    g, d, state = [nn.ParameterStore(seed) for seed in [3, 4, 0]]
    x, z = nn.zeros([2, 3]), nn.zeros([3, 2])
    with pytest.raises(RuntimeError): nn.SpectralGan(g, g, state, x, z, 3, 2)
    with pytest.raises(RuntimeError): nn.SpectralGan(g, d, state, nn.full([2, 3], math.nan), z, 3, 2)
    assert sum(p.size() for p in [g, d, state]) == 0
    nn.SpectralGan(g, d, state, x, z, 3, 2)
    empty = nn.ParameterStore(0)
    before = list(d.flatten())
    with pytest.raises(RuntimeError): nn.advance_cached_spectral_state(d, empty)
    assert list(d.flatten()) == before and empty.size() == 0
