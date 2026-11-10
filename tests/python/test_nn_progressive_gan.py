"""Native progressive paths, source activity, owning records and atomic phase captures."""
import gc
import math
import pytest
from pydatamunge import nn


def inputs():
    t = nn.DType.Float64
    p = [nn.ParameterStore(seed, t) for seed in [6401, 6407, 6413, 6421, 6427, 6449, 6451, 6469]]
    x = nn.full([2, 8, 8, 1], .3, t)
    z = nn.GaussianRejectionPrior(2, 16, 6504, 2, t).z
    a = nn.full([], .5, t)
    return p, x, z, a


def test_pixel_rms_division_and_higher_paths_without_batch_or_affine_state():
    x = nn.from_vector_double([3, 4, 0, 0], [2, 2]).requires_grad_()
    out = nn.pixel_rms_normalize(x)
    assert list(out.values()) == pytest.approx([.8485281371, 1.1313708499, 0, 0])
    options = nn.GradOptions()
    options.create_graph = True
    gradient = nn.grad(nn.sum(out), [x], options)[0]
    assert tuple(nn.grad(nn.sum(nn.mul(gradient, gradient)), [x])[0].shape()) == (2, 2)
    with pytest.raises(RuntimeError): nn.pixel_rms_normalize(x, 0)
    with pytest.raises(RuntimeError): nn.pixel_rms_normalize(nn.full([2, 2], math.nan))


def test_native_activity_returns_list_and_all_groups_initialize_with_both_alpha_forms():
    assert nn.ProgressiveActiveGroups(0) == [0, 1]
    assert nn.ProgressiveActiveGroups(1) == [0, 1, 2, 3]
    assert nn.ProgressiveActiveGroups(2, False) == [4, 6, 7]
    with pytest.raises(RuntimeError): nn.ProgressiveActiveGroups(3)
    p, x, z, a = inputs()
    one = nn.ProgressiveGan(*p, x, z, a)
    two = nn.ProgressiveGan(*p, x, z, nn.full([1], .5, nn.DType.Float64))
    assert list(one.fake.values()) == list(two.fake.values())
    assert [s.count() for s in p] == [1236, 5, 296, 5, 213, 8, 296, 8]
    assert sum(s.size() for s in p) == 24


def test_real_generator_and_discriminator_fade_endpoints_are_immutable():
    p, x, z, a = inputs()
    nn.ProgressiveGan(*p, x, z, a)
    before = [list(s.flatten()) for s in p]
    old = nn.ProgressiveGanGenerate(*p[:4], z, nn.full([], 0, nn.DType.Float64))
    new = nn.ProgressiveGanGenerate(*p[:4], z, nn.full([], 1, nn.DType.Float64))
    assert list(old.image.values()) == list(nn.upsample_nearest(old.old4, 2).values())
    assert list(new.image.values()) == list(new.new8.values())
    low = nn.ProgressiveGanDiscriminate(*p[4:], x, nn.full([], 0, nn.DType.Float64))
    high = nn.ProgressiveGanDiscriminate(*p[4:], x, nn.full([], 1, nn.DType.Float64))
    assert list(low.mixed.values()) == list(low.old_path.values())
    assert list(high.mixed.values()) == list(high.new_path.values())
    assert [list(s.flatten()) for s in p] == before


def test_capture_all_active_gradients_before_writes_then_fresh_generator_forward():
    p, x, z, a = inputs()
    nn.ProgressiveGan(*p, x, z, a)
    config = nn.OptimizerConfig()
    config.kind, config.beta1 = nn.OptimizerKind.Adam, .5
    optim = [nn.Optimizer(s, .0002, config) for s in p]
    options = nn.GradOptions()
    options.retain_graph, options.allow_unused = True, True
    for generator in [False, True]:
        groups = nn.ProgressiveActiveGroups(1, generator)
        out = nn.ProgressiveGan(*p, x, z, a)
        loss = out.g_loss if generator else out.d_loss
        captured = [nn.grad(loss, list(p[k].parameters()), options) for k in groups]
        before = [list(s.flatten()) for s in p]
        for k, gradient in zip(groups, captured): optim[k].step(gradient)
        after = [list(s.flatten()) for s in p]
        assert all(before[k] == after[k] for k in range(8) if k not in groups)
    assert [o.iterations() for o in optim] == [1]*8


def test_fields_and_live_noise_alpha_graphs_outlive_parents_and_stores():
    p, x, z, a = inputs()
    z.requires_grad_()
    a.requires_grad_()
    out = nn.ProgressiveGan(*p, x, z, a)
    features = nn.ProgressiveGanDiscriminate(*p[4:], x, a)
    fake, loss, mixed = out.fake, out.g_loss, features.mixed
    del out, features, p
    gc.collect()
    assert tuple(fake.shape()) == (2, 8, 8, 1) and tuple(mixed.shape()) == (2, 4, 4, 4)
    dz, da = nn.grad(loss, [z, a])
    assert any(abs(v) > 1e-12 for v in dz.values()) and abs(da.item()) > 1e-12


def test_source_G_loss_has_no_half_factor_and_invalid_checks_precede_leaves():
    p, x, z, a = inputs()
    with pytest.raises(RuntimeError): nn.ProgressiveGan(*p, x, z, nn.full([2], .5, nn.DType.Float64))
    with pytest.raises(RuntimeError): nn.ProgressiveGan(*p, x, z, nn.full([], 1.1, nn.DType.Float64))
    aliases = list(p)
    aliases[4] = aliases[0]
    with pytest.raises(RuntimeError): nn.ProgressiveGan(*aliases, x, z, a)
    assert sum(s.size() for s in p) == 0
    nn.ProgressiveGan(*p, x, z, a)
    for store in p[4:]:
        for name in store.names(): store.set(name, nn.zeros(list(store.at(name).shape()), nn.DType.Float64))
    out = nn.ProgressiveGan(*p, x, z, a)
    assert out.d_loss.item() == .5 and out.g_loss.item() == 1
