"""Native captures, rejection audit, nested owning fields and explicit BigGAN phase writes."""
import gc
import math
import pytest
from pydatamunge import nn


def test_captured_normalization_and_detached_moving_candidates_outlive_parents():
    t = nn.DType.Float64
    stats = nn.ParameterStore(0, t)
    x = nn.from_vector_double([1, 3], [2, 1]).requires_grad_()
    out = nn.capture_batch_norm(stats, 'bn', x, True)
    assert list(out.mean.values()) == [2] and list(out.unbiased_variance.values()) == [2]
    assert list(stats.at('bn/moving_mean').values()) == [0]
    value, mean, variance = out.value, out.mean, out.unbiased_variance
    loss = nn.add(nn.sum(value), nn.sum(mean))
    candidate = nn.batch_norm_moving_values(stats.at('bn/moving_mean'), stats.at('bn/moving_var'), mean, variance, .5)
    moving_mean, moving_variance = candidate.mean, candidate.variance
    del out, candidate, stats
    gc.collect()
    assert moving_mean.item() == 1 and moving_variance.item() == 1.5
    assert not moving_mean.requires_grad() and not moving_variance.requires_grad()
    assert list(nn.grad(loss, [x])[0].values()) == pytest.approx([.5, .5])
    with pytest.raises(RuntimeError): nn.capture_batch_norm(nn.ParameterStore(0), 'bn', nn.ones([1, 2]), True)


def test_rejection_prior_exact_candidates_casting_budget_and_owned_fields():
    a = nn.GaussianRejectionPrior(1, 3, 6907, .5)
    assert list(a.z.values()) == [-0.4454278349876404, -0.08479340374469757, 0.38326919078826904]
    assert list(a.proposals.values()) == [-0.9388149930815035, -0.44542782166946016, -1.2078747239081584, -0.08479340404496108, 0.3832691843572766]
    wide = nn.GaussianRejectionPrior(1, 3, 6907, .5, nn.DType.Float64)
    assert list(a.z.values()) == list(nn.cast(wide.z, nn.DType.Float32).values())
    z, proposals = a.z, a.proposals
    del a, wide
    gc.collect()
    assert not z.requires_grad() and not proposals.requires_grad()
    with pytest.raises(RuntimeError): nn.GaussianRejectionPrior(1, 1, 1, 1e-300)
    for limit in [0, -1, math.nan]:
        with pytest.raises(RuntimeError): nn.GaussianRejectionPrior(1, 3, 1, limit)


def test_census_display_switch_and_both_generator_loss_paths_are_explicit():
    stores = [nn.ParameterStore(seed) for seed in [6803, 6823, 0, 0, 0]]
    x, z, c = nn.full([2, 8, 8, 1], .3), nn.GaussianRejectionPrior(2, 12, 6872).z, nn.from_vector_int64([0, 1], [2])
    a = nn.BigGan(*stores, x, z, c)
    assert [p.count() for p in stores] == [789, 241, 241, 77, 24]
    assert [p.size() for p in stores] == [17, 9, 18, 10, 6]
    for p, s in [(stores[0], stores[2]), (stores[1], stores[3])]: nn.advance_cached_spectral_state(p, s, 16)
    before = [list(p.flatten()) for p in stores]
    a, b = nn.BigGan(*stores, x, z, c, False), nn.BigGan(*stores, x, z, c, True)
    assert before == [list(p.flatten()) for p in stores]
    assert list(a.fake.values()) == list(a.fake_inference.values())
    assert list(b.fake.values()) == list(b.fake_training.values())
    assert list(a.fake.values()) != list(b.fake.values())
    assert a.g_loss.item() == b.g_loss.item() and a.d_loss.item() == b.d_loss.item()
    assert tuple(a.moments.bn1_mean.shape()) == (4,)
    inference = nn.BigGanGenerate(stores[0], stores[2], stores[4], z, c)
    assert not inference.moments.bn1_mean.defined()


def test_five_phase_writes_and_atomic_captured_BN_commit_after_G_Adam():
    stores = [nn.ParameterStore(seed) for seed in [6803, 6823, 0, 0, 0]]
    g, d, gs, ds, moving = stores
    x, z, c = nn.full([2, 8, 8, 1], .3), nn.GaussianRejectionPrior(2, 12, 6872).z, nn.from_vector_int64([0, 1], [2])
    forward = lambda: nn.BigGan(*stores, x, z, c, True)
    forward()
    nn.advance_cached_spectral_state(g, gs, 16)
    nn.advance_cached_spectral_state(d, ds, 16)
    config = nn.OptimizerConfig()
    config.kind, config.beta1, config.beta2 = nn.OptimizerKind.Adam, 0, .999
    go, disc = nn.Optimizer(g, .0002, config), nn.Optimizer(d, .0002, config)
    captured = None

    def g_write():
        nonlocal captured
        out = forward()
        captured = out.moments
        go.step(nn.grad(out.g_loss, list(g.parameters())))

    actions = [(3, lambda: nn.advance_cached_spectral_state(d, ds)),
               (1, lambda: disc.step(nn.grad(forward().d_loss, list(d.parameters())))),
               (2, lambda: nn.advance_cached_spectral_state(g, gs)), (0, g_write),
               (4, lambda: nn.BigGanCommitMoments(moving, captured))]
    for owner, action in actions:
        before = [list(p.flatten()) for p in stores]
        action()
        after = [list(p.flatten()) for p in stores]
        assert before[owner] != after[owner]
        assert all(before[i] == after[i] for i in range(5) if i != owner)
    assert go.iterations() == disc.iterations() == 1
    before = list(moving.flatten())
    captured.output_unbiased_variance = nn.full([4], math.nan)
    with pytest.raises(RuntimeError): nn.BigGanCommitMoments(moving, captured)
    assert list(moving.flatten()) == before


def test_nested_moments_and_projection_fields_retain_storage_and_graph_after_collection():
    t = nn.DType.Float64
    stores = [nn.ParameterStore(seed, t) for seed in [6803, 6823, 0, 0, 0]]
    x, z, c = nn.full([2, 8, 8, 1], .3, t), nn.GaussianRejectionPrior(2, 12, 6872, 2., t).z.requires_grad_(), nn.from_vector_int64([0, 1], [2])
    nn.BigGan(*stores, x, z, c)
    nn.advance_cached_spectral_state(stores[0], stores[2], 16)
    nn.advance_cached_spectral_state(stores[1], stores[3], 16)
    out = nn.BigGan(*stores, x, z, c, True)
    generated = nn.BigGanGenerate(stores[0], stores[2], stores[4], z, c, True)
    discriminated = nn.BigGanDiscriminate(stores[1], stores[3], x, c)
    capture, image, projection = out.moments, generated.image, discriminated.projection
    loss = nn.add(out.g_loss, nn.sum(capture.bn1_mean))
    del out, generated, discriminated, stores
    gc.collect()
    assert tuple(image.shape()) == (2, 8, 8, 1) and tuple(projection.shape()) == (2, 1)
    assert tuple(capture.bn1_mean.shape()) == (4,)
    assert any(abs(v) > 1e-10 for v in nn.grad(loss, [z])[0].values())


def test_projection_orthogonal_resize_and_invalid_inputs_precede_leaves():
    p = nn.ParameterStore(1, nn.DType.Float64)
    p.set('layer/W', nn.ones([2, 2], nn.DType.Float64))
    p.set('bias', nn.zeros([2], nn.DType.Float64))
    penalty = nn.ModifiedOrthogonalPenalty(p, .5)
    assert penalty.item() == 4
    partials = nn.grad(penalty, list(p.parameters()))
    assert list(partials[0].values()) == [4, 4, 4, 4] and list(partials[1].values()) == [0, 0]
    assert tuple(nn.BigGanResize8(nn.ones([2, 28, 28, 1])).shape()) == (2, 8, 8, 1)
    g, d, gs, ds, moving = [nn.ParameterStore(seed) for seed in [1, 2, 0, 0, 0]]
    x, z, c = nn.zeros([2, 8, 8, 1]), nn.zeros([2, 12]), nn.from_vector_int64([0, 1], [2])
    with pytest.raises(RuntimeError): nn.BigGan(g, g, gs, ds, moving, x, z, c)
    with pytest.raises(RuntimeError): nn.BigGan(g, d, gs, ds, moving, x, nn.zeros([3, 12]), c)
    with pytest.raises(RuntimeError): nn.BigGan(g, d, gs, ds, moving, x, z, nn.from_vector_int64([0, 10], [2]))
    assert sum(p.size() for p in [g, d, gs, ds, moving]) == 0
