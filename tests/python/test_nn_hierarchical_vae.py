"""Native hierarchy/residual Gaussians, explicit BN state, ancestral decode and host Adam."""
import gc
import math
import pytest
from pydatamunge import nn


def test_residual_posterior_owns_fields_and_all_four_conditional_kl_paths_are_live():
    t = nn.DType.Float64
    prior_mu = nn.full([2, 3], .4, t).requires_grad_()
    prior_lv = nn.full([2, 3], -.2, t).requires_grad_()
    zero = nn.zeros([2, 3], t)
    posterior = nn.HierarchyPosterior(prior_mu, prior_lv, zero, zero)
    mu, lv = posterior.mu, posterior.log_variance
    del posterior
    gc.collect()
    assert list(mu.values()) == list(prior_mu.values())
    assert list(lv.values()) == list(prior_lv.values())
    assert list(nn.ConditionalGaussianKl(mu, lv, prior_mu, prior_lv).values()) == [0, 0]
    qmu = nn.full([2, 3], -.3, t).requires_grad_()
    qlv = nn.full([2, 3], .1, t).requires_grad_()
    loss = nn.sum(nn.ConditionalGaussianKl(qmu, qlv, prior_mu, prior_lv))
    gradients = nn.grad(loss, [qmu, qlv, prior_mu, prior_lv])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_all_nineteen_fields_survive_collection_and_both_latents_decode_without_image():
    t = nn.DType.Float64
    for nvae in [False, True]:
        p, state = nn.ParameterStore(2609, t), nn.ParameterStore(0, t)
        out = nn.HierarchicalVariationalAutoencoder(p, state, nn.full([2, 16], .3, t), nn.ones([2, 2], t), nvae, False, False, 4, 4, 1, 1)
        names = ['encoder_feature', 'top_feature', 'top_mu', 'top_lv', 'top_z', 'prior_mu', 'prior_lv',
                 'head_mu', 'head_lv', 'lower_mu', 'lower_lv', 'lower_z', 'probability', 'nll',
                 'kl_lower', 'kl_top', 'loss', 'regularizer', 'train_loss']
        fields = {name: getattr(out, name) for name in names}
        del out
        gc.collect()
        assert p.size() == (37 if nvae else 26) and p.count() == (96 if nvae else 70)
        assert state.size() == (6 if nvae else 0)
        assert tuple(fields['encoder_feature'].shape()) == (2, 1, 1, 1)
        assert tuple(fields['probability'].shape()) == (2, 16)
        assert fields['train_loss'].item() == fields['loss'].item() + fields['regularizer'].item()
        decoded = nn.HierarchyDecode(p, state, fields['lower_z'], fields['top_z'], nvae, 4, 4, 1)
        assert list(decoded.values()) == list(fields['probability'].values())
        gradients = nn.grad(fields['train_loss'], list(p.parameters()))
        assert len(gradients) == p.size() and all(all(math.isfinite(v) for v in g.values()) for g in gradients)


def test_training_moments_and_updates_are_explicit_and_optimizer_excludes_state():
    t = nn.DType.Float64
    p, state = nn.ParameterStore(2609, t), nn.ParameterStore(0, t)
    x, epsilon = nn.full([2, 16], .3, t), nn.from_vector_double([.3, -.6, -.2, .8], [2, 2])
    nn.HierarchicalVariationalAutoencoder(p, state, x, epsilon, True, False, False, 4, 4, 1, 1)
    weights, before = tuple(p.flatten()), tuple(state.flatten())
    probe = nn.HierarchicalVariationalAutoencoder(p, state, x, epsilon, True, True, False, 4, 4, 1, 1)
    assert tuple(state.flatten()) == before
    update = nn.HierarchicalVariationalAutoencoder(p, state, x, epsilon, True, True, True, 4, 4, 1, 1)
    assert tuple(p.flatten()) == weights and tuple(state.flatten()) != before
    assert list(probe.probability.values()) == list(update.probability.values())
    stats = tuple(state.flatten())
    gradients = nn.grad(update.train_loss, list(p.parameters()))
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != weights
    assert tuple(state.flatten()) == stats


def test_default_census_ancestral_fields_and_float64_non_square_geometry():
    for nvae in [False, True]:
        p, state = nn.ParameterStore(2609), nn.ParameterStore(0)
        nn.HierarchicalVariationalAutoencoder(p, state, nn.zeros([2, 784]), nn.zeros([2, 51]), nvae)
        assert p.count() == (2027 if nvae else 1977) and p.size() == (37 if nvae else 26)
        weights, stats = tuple(p.flatten()), tuple(state.flatten())
        prior = nn.HierarchyPrior(p, state, nn.ones([2, 2]), nn.zeros([2, 49]), nvae)
        top, z, pm, pl, epsilon, probability = prior.top_z, prior.lower_z, prior.prior_mu, prior.prior_lv, prior.lower_noise, prior.probability
        del prior
        gc.collect()
        assert list(z.values()) == list(pm.values())
        assert tuple(pl.shape()) == tuple(epsilon.shape()) == (2, 49)
        assert list(probability.values()) == list(nn.HierarchyDecode(p, state, z, top, nvae).values())
        assert tuple(p.flatten()) == weights and tuple(state.flatten()) == stats
    t = nn.DType.Float64
    p, state = nn.ParameterStore(2609, t), nn.ParameterStore(0, t)
    out = nn.HierarchicalVariationalAutoencoder(p, state, nn.full([2, 32], .3, t), nn.ones([2, 5], t), True, False, False, 4, 8, 2, 3)
    assert tuple(out.probability.shape()) == (2, 32) and tuple(out.top_mu.shape()) == (2, 3)
    assert tuple(out.lower_mu.shape()) == (2, 2) and out.loss.dtype() == t
    cell, buffers = nn.ParameterStore(2609, t), nn.ParameterStore(0, t)
    feature = nn.HierarchyNvaeCell(cell, buffers, nn.ones([2, 2, 2, 4], t), True, False)
    assert tuple(feature.shape()) == (2, 2, 2, 4) and cell.count() == 198 and buffers.count() == 40


def test_invalid_stores_modes_pixels_geometry_and_noise_fail_before_creation():
    p, state = nn.ParameterStore(1), nn.ParameterStore(0)
    x, noise = nn.zeros([2, 16]), nn.zeros([2, 2])
    for stats, training, update, height in [(p, False, False, 4), (state, False, True, 4), (state, False, False, 5)]:
        with pytest.raises(RuntimeError):
            nn.HierarchicalVariationalAutoencoder(p, stats, x, noise, True, training, update, height, 4, 1, 1)
    with pytest.raises(RuntimeError):
        nn.HierarchicalVariationalAutoencoder(p, state, nn.full([2, 16], 1.1), noise, False, False, False, 4, 4, 1, 1)
    with pytest.raises(RuntimeError):
        nn.HierarchyPrior(p, state, nn.zeros([2, 1]), nn.zeros([1, 1]), False, 4, 4, 1)
    with pytest.raises(RuntimeError):
        nn.HierarchyNvaeCell(p, state, nn.zeros([2, 2, 2, 1]), False, True)
    assert p.size() == state.size() == 0
