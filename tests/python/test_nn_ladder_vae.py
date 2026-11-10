"""Native precision fusion, owning Ladder endpoints and explicit two-level noise."""
import gc
import math
import pytest
from pydatamunge import nn


def test_fused_gaussian_fields_survive_collection_and_all_four_paths_remain_live():
    t = nn.DType.Float64
    em, el = nn.full([2, 2], 2, t).requires_grad_(), nn.full([2, 2], math.log(4), t).requires_grad_()
    pm, pl = nn.full([2, 2], -1, t).requires_grad_(), nn.full([2, 2], math.log(9), t).requires_grad_()
    q = nn.LadderPrecisionFusion(em, el, pm, pl)
    mu, lv = q.mu, q.log_variance
    del q
    gc.collect()
    assert list(mu.values()) == pytest.approx([14/13]*4)
    assert [math.exp(v) for v in lv.values()] == pytest.approx([36/13]*4)
    loss = nn.add(nn.sum(nn.add(nn.square(mu), nn.square(lv))), nn.sum(nn.ConditionalGaussianKl(mu, lv, pm, pl)))
    gradients = nn.grad(loss, [em, el, pm, pl])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_all_sixteen_fields_live_float64_inputs_and_lower_only_decode():
    t = nn.DType.Float64
    p = nn.ParameterStore(2503, t)
    image, noise = nn.full([2, 6], .3, t).requires_grad_(), nn.full([2, 3], .2, t).requires_grad_()
    out = nn.LadderVariationalAutoencoder(p, image, noise, 6, 2, 1, 3, 2)
    names = ['evidence_mu', 'evidence_lv', 'top_mu', 'top_lv', 'top_z', 'prior_mu', 'prior_lv',
             'lower_mu', 'lower_lv', 'lower_z', 'logits', 'probability', 'nll', 'kl_lower', 'kl_top', 'loss']
    fields = {name: getattr(out, name) for name in names}
    del out
    gc.collect()
    assert p.size() == 22 and p.count() == 100
    assert tuple(fields['probability'].shape()) == (2, 6) and tuple(fields['top_z'].shape()) == (2, 1)
    assert fields['loss'].dtype() == t
    before = tuple(p.flatten())
    decoded = nn.sigmoid(nn.GaussianVaeDecode(p, fields['lower_z'], 6, 2, 3))
    assert list(decoded.values()) == list(fields['probability'].values())
    assert list(nn.sigmoid(fields['logits']).values()) == list(decoded.values())
    assert tuple(p.flatten()) == before
    gradients = nn.grad(fields['loss'], list(p.parameters()) + [image, noise])
    assert len(gradients) == 24 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients[-2:])


def test_default_census_noise_order_and_visible_adam():
    p = nn.ParameterStore(2503)
    image, zero = nn.full([2, 784], .3), nn.zeros([2, 6])
    base = nn.LadderVariationalAutoencoder(p, image, zero)
    assert p.size() == 22 and p.count() == 52196
    assert list(base.lower_z.values()) == list(base.lower_mu.values())
    assert list(base.top_z.values()) == list(base.top_mu.values())
    lower = nn.concat([nn.full([2, 4], .5), nn.zeros([2, 2])], 1)
    upper = nn.concat([nn.zeros([2, 4]), nn.full([2, 2], .5)], 1)
    low, top = nn.LadderVariationalAutoencoder(p, image, lower), nn.LadderVariationalAutoencoder(p, image, upper)
    assert list(low.lower_mu.values()) == list(base.lower_mu.values())
    assert list(low.prior_mu.values()) == list(base.prior_mu.values())
    assert list(low.lower_z.values()) != list(base.lower_z.values())
    assert list(top.prior_mu.values()) != list(base.prior_mu.values())
    assert list(top.lower_mu.values()) != list(base.lower_mu.values())
    before = tuple(p.flatten())
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    optimizer.step(nn.grad(top.loss, list(p.parameters())))
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_ancestral_fields_survive_collection_and_both_explicit_inputs_have_gradients():
    t = nn.DType.Float64
    p = nn.ParameterStore(2503, t)
    top, epsilon = nn.full([2, 2], .2, t).requires_grad_(), nn.ones([2, 4], t).requires_grad_()
    prior = nn.LadderPrior(p, top, epsilon)
    names = ['top_z', 'lower_noise', 'prior_mu', 'prior_lv', 'lower_z', 'probability']
    fields = {name: getattr(prior, name) for name in names}
    del prior
    gc.collect()
    assert p.size() == 10 and p.count() == 26216
    assert list(fields['top_z'].values()) == list(top.values())
    assert list(fields['lower_noise'].values()) == list(epsilon.values())
    assert list(nn.sigmoid(nn.GaussianVaeDecode(p, fields['lower_z'])).values()) == list(fields['probability'].values())
    gradients = nn.grad(nn.sum(fields['probability']), [top, epsilon])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_invalid_shapes_pixels_types_and_widths_fail_before_creation():
    p = nn.ParameterStore(1)
    x, e = nn.full([2, 6], .3), nn.zeros([2, 3])
    with pytest.raises(RuntimeError):
        nn.LadderVariationalAutoencoder(p, x, e, 6, 2, 1, 0, 2)
    with pytest.raises(RuntimeError):
        nn.LadderVariationalAutoencoder(p, nn.full([2, 6], 1.1), e, 6, 2, 1, 3, 2)
    with pytest.raises(RuntimeError):
        nn.LadderVariationalAutoencoder(p, x, nn.zeros([1, 3]), 6, 2, 1, 3, 2)
    with pytest.raises(RuntimeError):
        nn.LadderPrior(p, nn.zeros([2, 1]), nn.zeros([1, 2]), 6, 2, 1, 3, 2)
    assert p.size() == 0
    with pytest.raises(RuntimeError):
        nn.LadderPrecisionFusion(nn.zeros([2, 2]), nn.zeros([1, 2]), nn.zeros([2, 2]), nn.zeros([2, 2]))
    with pytest.raises(RuntimeError):
        nn.LadderPrecisionFusion(nn.zeros([2, 2]), nn.full([2, 2], float('inf')), nn.zeros([2, 2]), nn.zeros([2, 2]))
