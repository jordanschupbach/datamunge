"""Native fused captures, exact phase conventions and host-owned DCGAN writes."""
import gc
import math
import pytest
from pydatamunge import nn


def bank(dtype=None):
    dtype = nn.DType.Float32 if dtype is None else dtype
    stores = [nn.ParameterStore(seed, dtype) for seed in [5003, 5011, 0, 0]]
    x = nn.full([2, 28, 28, 1], .3, dtype)
    z = nn.GaussianRejectionPrior(2, 16, 4862, 2, dtype).z
    return stores, x, z


def test_fused_affine_capture_owns_detached_unbiased_moments_and_live_value():
    t = nn.DType.Float64
    p, s = nn.ParameterStore(0, t), nn.ParameterStore(0, t)
    x = nn.from_vector_double([1, 3], [2, 1]).requires_grad_()
    out = nn.capture_affine_batch_norm(p, s, 'bn', x, True)
    assert out.mean.item() == 2 and out.unbiased_variance.item() == 2
    assert not out.mean.requires_grad() and not out.unbiased_variance.requires_grad()
    assert list(s.at('bn/moving_mean').values()) == [0]
    next = nn.batch_norm_moving_values(s.at('bn/moving_mean'), s.at('bn/moving_var'), out.mean, out.unbiased_variance, .9)
    assert next.mean.item() == pytest.approx(.2) and next.variance.item() == pytest.approx(1.1)
    value, mean, variance = out.value, out.mean, out.unbiased_variance
    del out, p, s
    gc.collect()
    assert mean.item() == variance.item() == 2
    assert len(nn.grad(nn.sum(nn.mul(value, value)), [x])[0].values()) == 2


def test_saved_inference_census_and_generation_conversion_are_explicit():
    stores, x, z = bank()
    out = nn.Dcgan(*stores, x, z)
    assert [s.count() for s in stores] == [6873, 989, 24, 16]
    assert [s.size() for s in stores] == [8, 7, 4, 2]
    before = [list(s.flatten()) for s in stores]
    generated = nn.DcganGenerate(stores[0], stores[2], z)
    assert not generated.stem_mean.defined()
    pixels = nn.reshape(nn.mul(nn.add(generated.image, 1), .5), [2, 784])
    assert list(out.fake.values()) == list(pixels.values())
    assert before == [list(s.flatten()) for s in stores]
    assert tuple(out.fake.shape()) == (2, 784)


def test_phase_modes_stop_D_fake_and_leave_G_group_selection_to_caller():
    stores, x, z = bank(nn.DType.Float64)
    z.requires_grad_()
    a, b = nn.DcganDiscriminatorPhase(*stores, x, z), nn.DcganGeneratorPhase(*stores, x, z)
    assert not a.fake.requires_grad() and b.fake.requires_grad()
    assert not a.down7_mean.requires_grad() and not a.stem_mean.requires_grad()
    assert not b.real_logit.defined() and not b.down7_mean.defined()
    partials = nn.grad(b.loss, [z]+list(stores[1].parameters()))
    assert any(abs(v) > 1e-10 for v in partials[0].values())
    assert len(partials) == 8
    direct = nn.DcganDiscriminate(stores[1], stores[3], b.fake)
    assert list(b.fake_logit.values()) == list(direct.logit.values())
    assert b.loss.item() == nn.mean(nn.softplus(nn.neg(direct.logit))).item()


def test_four_explicit_phase_owners_commit_captures_before_selected_Adam():
    stores, x, z = bank()
    g, d, gs, ds = stores
    nn.Dcgan(*stores, x, z)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    config.beta1 = .5
    go, disc = nn.Optimizer(g, .0002, config), nn.Optimizer(d, .0002, config)

    def commit(state, layer, mean, variance):
        next = nn.batch_norm_moving_values(state.at(layer+'/moving_mean'), state.at(layer+'/moving_var'), mean, variance, .9)
        state.set(layer+'/moving_mean', next.mean)
        state.set(layer+'/moving_var', next.variance)

    a = nn.DcganDiscriminatorPhase(*stores, x, z)
    actions = [(3, lambda: commit(ds, 'down7_bn', a.down7_mean, a.down7_variance)),
               (1, lambda: disc.step(nn.grad(a.loss, list(d.parameters()))))]
    for owner, action in actions:
        before = [list(s.flatten()) for s in stores]
        action()
        after = [list(s.flatten()) for s in stores]
        assert before[owner] != after[owner]
        assert all(before[i] == after[i] for i in range(4) if i != owner)
    b = nn.DcganGeneratorPhase(*stores, x, z)
    def g_commit():
        commit(gs, 'stem_bn', b.stem_mean, b.stem_variance)
        commit(gs, 'up14_bn', b.up14_mean, b.up14_variance)
    for owner, action in [(2, g_commit), (0, lambda: go.step(nn.grad(b.loss, list(g.parameters()))))]:
        before = [list(s.flatten()) for s in stores]
        action()
        after = [list(s.flatten()) for s in stores]
        assert before[owner] != after[owner]
        assert all(before[i] == after[i] for i in range(4) if i != owner)
    assert go.iterations() == disc.iterations() == 1


def test_flat_record_fields_and_graphs_outlive_parents_and_invalid_inputs_precede_leaves():
    stores, x, z = bank(nn.DType.Float64)
    z.requires_grad_()
    out = nn.DcganGeneratorPhase(*stores, x, z)
    fake, loss, mean = out.fake, out.loss, out.stem_mean
    del out, stores
    gc.collect()
    assert tuple(fake.shape()) == (2, 28, 28, 1) and tuple(mean.shape()) == (8,)
    assert not mean.requires_grad()
    assert any(abs(v) > 1e-10 for v in nn.grad(loss, [z])[0].values())
    stores, x, z = bank()
    with pytest.raises(RuntimeError): nn.Dcgan(stores[0], stores[0], stores[2], stores[3], x, z)
    with pytest.raises(RuntimeError): nn.Dcgan(*stores, x, nn.ones([1, 16]))
    with pytest.raises(RuntimeError): nn.DcganGenerate(stores[0], stores[2], nn.full([2, 16], math.nan))
    assert sum(s.size() for s in stores) == 0
