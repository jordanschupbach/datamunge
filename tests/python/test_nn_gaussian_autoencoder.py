"""Canonical Gaussian VAE endpoints, explicit gradients and fitting-only factor probes."""
import gc
import math
import pytest
from pydatamunge import nn


def test_vanilla_configuration_compatibility_fields_ownership_and_caller_adam():
    from pydatamunge import nn_components
    t = nn.DType.Float64
    p = nn.ParameterStore(1951, t)
    image = nn.full([2, 784], .25, t).requires_grad_()
    noise = nn.full([2, 2], .2, t).requires_grad_()
    canonical = nn.BetaVariationalAutoencoder(p, image, noise, 1, 784, 2, 64)
    compatibility = nn_components.ae.VariationalAutoencoder(p, image, noise)
    names = ['mu', 'log_variance', 'z', 'logits', 'probability', 'reconstruction_nll', 'kl', 'loss']
    fields = {name: getattr(compatibility, name) for name in names}
    del compatibility
    gc.collect()
    assert p.size() == 10 and p.count() == 101652
    assert all(list(fields[name].values()) == list(getattr(canonical, name).values()) for name in names)
    assert list(nn.sigmoid(nn.GaussianVaeDecode(p, fields['z'], 784, 2, 64)).values()) == list(fields['probability'].values())
    gradients = nn.grad(canonical.loss, list(p.parameters()) + [image, noise])
    assert len(gradients) == 12 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients[-2:])
    before = tuple(p.flatten())
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    optimizer.step(gradients[:10])
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_explicit_pathwise_sample_kl_and_all_input_derivatives():
    mu = nn.from_vector_double([.2, -.4, .1, .3], [2, 2]).requires_grad_()
    lv = nn.from_vector_double([0, math.log(4), math.log(.25), .2], [2, 2]).requires_grad_()
    eps = nn.from_vector_double([1, -.5, 2, -1], [2, 2]).requires_grad_()
    z, kl = nn.GaussianReparameterize(mu, lv, eps), nn.DiagonalGaussianKl(mu, lv)
    loss = nn.mean(nn.add(nn.mul(nn.sum(nn.square(z), [1]), .5), nn.mul(kl, 4)))
    gradients = nn.grad(loss, [mu, lv, eps])
    for i, (m, l, e) in enumerate(zip(mu.values(), lv.values(), eps.values())):
        sigma = math.exp(.5*l)
        value = m + sigma*e
        assert list(z.values())[i] == pytest.approx(value, abs=1e-14)
        assert list(gradients[0].values())[i] == pytest.approx((value+4*m)/2, abs=1e-14)
        assert list(gradients[1].values())[i] == pytest.approx((.5*value*sigma*e+2*(math.exp(l)-1))/2, abs=1e-14)
        assert list(gradients[2].values())[i] == pytest.approx(value*sigma/2, abs=1e-14)


def test_gaussian_result_lifetime_paired_beta_and_host_owned_adam():
    p = nn.ParameterStore(2237)
    x, epsilon = nn.full([2, 3], .3), nn.full([2, 2], .7)
    out = nn.BetaVariationalAutoencoder(p, x, epsilon, 4, 3, 2, 2)
    assert p.size() == 10 and p.count() == 35
    probability, mu, loss = out.probability, out.mu, out.loss
    expected = tuple(probability.values())
    del out
    gc.collect()
    one = nn.BetaVariationalAutoencoder(p, x, epsilon, 1, 3, 2, 2)
    assert tuple(one.probability.values()) == expected
    assert tuple(one.mu.values()) == tuple(mu.values())
    assert loss.item() - one.loss.item() == pytest.approx(3*nn.mean(one.kl).item(), abs=2e-6)
    assert tuple(nn.sigmoid(nn.GaussianVaeDecode(p, one.z, 3, 2, 2)).values()) == expected
    gradients = nn.grad(loss, list(p.parameters()))
    assert len(gradients) == 10
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_fit_only_generic_probe_owns_thirteen_detached_arrays():
    fit = nn.from_vector_double([-1, -1, 0, 0, 1, 1], [3, 2])
    test = nn.from_vector_double([2, 2, 3, 3], [2, 2])
    y = nn.from_vector_double([1, 3, 5], [3, 1])
    yt = nn.from_vector_double([7, 9], [2, 1])
    lv = nn.zeros([3, 2], nn.DType.Float64)
    first = dict(nn.LatentFactorDiagnostics(fit, test, y, yt, lv))
    second = dict(nn.LatentFactorDiagnostics(fit, test, y, nn.from_vector_double([-100, -99], [2, 1]), lv))
    assert len(first) == 13 and tuple(first['best_axis_by_fit'].values()) == (0,)
    for name in ['latent_mean', 'latent_std', 'factor_mean', 'ridge_coefficients', 'best_axis_by_fit', 'ridge_prediction', 'single_prediction']:
        assert tuple(first[name].values()) == tuple(second[name].values())
    assert first['ridge_r2'].item() > .99999 and second['ridge_r2'].item() < 0
    prediction = first['ridge_prediction']
    del first
    gc.collect()
    assert tuple(prediction.shape()) == (2, 1) and not prediction.requires_grad()


def test_invalid_beta_shape_and_target_validation_before_leaf_creation():
    p = nn.ParameterStore(1)
    for beta in [0, -1, float('inf'), float('nan'), 1e-100, 1e100]:
        with pytest.raises(RuntimeError):
            nn.BetaVariationalAutoencoder(p, nn.zeros([2, 3]), nn.zeros([2, 2]), beta, 3, 2, 2)
    with pytest.raises(RuntimeError):
        nn.BetaVariationalAutoencoder(p, nn.zeros([2, 3]), nn.zeros([1, 2]), 1, 3, 2, 2)
    assert p.size() == 0
