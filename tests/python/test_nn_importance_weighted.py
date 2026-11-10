"""Native IWAE sampled-density bounds, owning diagnostics and host-owned updates."""
import gc
import math
import pytest
from pydatamunge import nn


def test_shifted_extremes_and_all_bound_fields_survive_collection():
    t = nn.DType.Float64
    w = nn.from_vector_double([1000, 1001, 999], [3, 1]).requires_grad_()
    bound = nn.IwaeLogMeanExp(w)
    assert bound.item() == pytest.approx(1001 + math.log((math.exp(-1) + 1 + math.exp(-2))/3), abs=1e-12)
    partials = nn.grad(nn.neg(nn.sum(bound)), [w])[0]
    assert sum(partials.values()) == pytest.approx(-1)
    mu, lv, z = nn.full([6, 2], .3, t).requires_grad_(), nn.full([6, 2], -.2, t).requires_grad_(), nn.ones([6, 2], t).requires_grad_()
    nll = nn.full([6], 3, t).requires_grad_()
    out = nn.ImportanceWeightedBound(mu, lv, z, nll, 3)
    fields = {name: getattr(out, name) for name in ['log_weights', 'normalized_weights', 'negative_bound', 'effective_samples', 'loss']}
    del out
    gc.collect()
    assert tuple(fields['log_weights'].shape()) == (3, 2)
    assert list(fields['effective_samples'].values()) == pytest.approx([3, 3])
    gradients = nn.grad(fields['loss'], [mu, lv, z, nll])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)
    assert sum(gradients[3].values()) == pytest.approx(1)


def test_all_eleven_forward_fields_live_float64_paths_and_image_free_decode():
    t = nn.DType.Float64
    p = nn.ParameterStore(2237, t)
    x, e = nn.full([2, 6], .3, t).requires_grad_(), nn.full([3, 2, 2], .2, t).requires_grad_()
    out = nn.ImportanceWeightedAutoencoder(p, x, e, 6, 2, 3)
    names = ['mu', 'log_variance', 'z', 'probability', 'nll', 'kl', 'loss', 'negative_bound', 'log_weights', 'normalized_weights', 'effective_samples']
    fields = {name: getattr(out, name) for name in names}
    del out
    gc.collect()
    assert p.size() == 10 and p.count() == 70
    assert tuple(fields['probability'].shape()) == (6, 6)
    assert tuple(fields['negative_bound'].shape()) == (2,)
    assert fields['loss'].dtype() == t
    before = tuple(p.flatten())
    decoded = nn.sigmoid(nn.GaussianVaeDecode(p, fields['z'], 6, 2, 3))
    assert list(decoded.values()) == list(fields['probability'].values())
    assert tuple(p.flatten()) == before
    gradients = nn.grad(fields['loss'], list(p.parameters()) + [x, e])
    assert len(gradients) == 12 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients[-2:])


def test_default_census_sampled_k1_and_visible_optimizer():
    p = nn.ParameterStore(2237)
    image, epsilon = nn.full([2, 784], .3), nn.full([1, 2, 4], .2)
    out = nn.ImportanceWeightedAutoencoder(p, image, epsilon)
    assert p.size() == 10 and p.count() == 51416
    plain = nn.BetaVariationalAutoencoder(p, image, nn.reshape(epsilon, [2, 4]))
    assert list(out.probability.values()) == list(plain.probability.values())
    assert out.loss.item() != plain.loss.item()
    before = tuple(p.flatten())
    gradients = nn.grad(out.loss, list(p.parameters()))
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_sample_major_layout_uses_shared_gaussian_leaves_and_weight_axis():
    p = nn.ParameterStore(2237)
    x = nn.full([2, 6], .3)
    e = nn.from_vector([.1, .3, -.2, .4, .5, -.1, .3, .2, -.4, .6, .2, -.5], [3, 2, 2])
    out = nn.ImportanceWeightedAutoencoder(p, x, e, 6, 2, 3)
    tiled = nn.reshape(nn.tile(nn.reshape(x, [1, 2, 6]), [3, 1, 1]), [6, 6])
    plain = nn.BetaVariationalAutoencoder(p, tiled, nn.reshape(e, [6, 2]), 1, 6, 2, 3)
    assert list(out.z.values()) == list(plain.z.values())
    assert list(out.probability.values()) == list(plain.probability.values())
    weights = list(out.normalized_weights.values())
    assert [sum(weights[b::2]) for b in range(2)] == pytest.approx([1, 1], abs=1e-6)
    assert all(1 - 1e-6 <= v <= 3 + 1e-6 for v in out.effective_samples.values())


def test_invalid_dimensions_dtype_values_and_pixels_fail_before_creation():
    p = nn.ParameterStore(1)
    x = nn.full([2, 6], .3)
    for noise in [nn.ones([3, 1, 2]), nn.ones([3, 2, 2], nn.DType.Float64), nn.full([3, 2, 2], float('nan'))]:
        with pytest.raises(RuntimeError):
            nn.ImportanceWeightedAutoencoder(p, x, noise, 6, 2, 3)
    with pytest.raises(RuntimeError):
        nn.ImportanceWeightedAutoencoder(p, nn.full([2, 6], 1.1), nn.ones([3, 2, 2]), 6, 2, 3)
    assert p.size() == 0
    for weights in [nn.zeros([0, 2]), nn.ones([2, 2], nn.DType.Int32), nn.full([2, 2], float('inf'))]:
        with pytest.raises(RuntimeError):
            nn.IwaeLogMeanExp(weights)
    with pytest.raises(RuntimeError):
        nn.ImportanceWeightedBound(nn.zeros([6, 2]), nn.zeros([6, 2]), nn.ones([6, 2]), nn.ones([6]), 4)
