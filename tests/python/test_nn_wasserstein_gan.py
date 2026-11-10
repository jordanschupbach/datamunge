"""Native WGAN input-Jacobian paths, audited priors, clipping and explicit host writes."""
import gc
import math
import pytest
from pydatamunge import nn


def inputs():
    t = nn.DType.Float64
    g, c = nn.ParameterStore(5501, t), nn.ParameterStore(5503, t)
    x = nn.from_vector_double([.2, .7, -.3, .8], [2, 2]).requires_grad_()
    z = nn.GaussianRejectionPrior(2, 2, 5601, 2, t).z.requires_grad_()
    a = nn.MidpointUniform(2, 1, 6601, t).requires_grad_()
    return g, c, x, z, a


def test_midpoint_uniform_goldens_final_cast_and_invalid_geometry():
    wide = nn.MidpointUniform(3, 1, 6601, nn.DType.Float64)
    assert list(wide.values()) == [0.14499446202535182, 0.4169559587026015, 0.9257562280399725]
    assert list(nn.MidpointUniform(3, 1, 6601).values()) == list(nn.cast(wide, nn.DType.Float32).values())
    assert not wide.requires_grad()
    with pytest.raises(RuntimeError): nn.MidpointUniform(0)
    with pytest.raises(RuntimeError): nn.MidpointUniform(2, 0)
    with pytest.raises(RuntimeError): nn.MidpointUniform(2, 1, 1, nn.DType.Int64)


def test_wasserstein_census_actual_jacobians_and_stopped_external_penalty_paths():
    g, c, x, z, a = inputs()
    out = nn.WassersteinGan(g, c, x, z, a)
    assert g.count() == 82 and c.count() == 65 and g.size() == c.size() == 4
    assert list(out.input_gradient.values()) == pytest.approx(list(out.symbolic_input_gradient.values()), abs=1e-14)
    assert not out.interpolation.requires_grad() and out.symbolic_input_gradient.requires_grad()
    options = nn.GradOptions()
    options.allow_unused = True
    stopped = nn.grad(out.penalty, [x, z, a, g.at('hidden/W')], options)
    assert all(not p.defined() for p in stopped)
    out = nn.WassersteinGan(g, c, x, z, a)
    options.create_graph = True
    partials = nn.grad(out.penalty, list(c.parameters()), options)
    assert partials[0].requires_grad() and not partials[-1].defined()
    assert tuple(nn.grad(nn.sum(nn.mul(partials[0], partials[0])), [c.at('hidden/W')])[0].shape()) == (2, 16)


def test_evaluation_no_grad_graph_flag_and_weight_zero_are_explicit():
    g, c, x, z, a = inputs()
    with nn.no_grad():
        out = nn.WassersteinGan(g, c, x, z, a, 10, False)
        assert all(not v.requires_grad() for v in [out.fake, out.input_gradient, out.symbolic_input_gradient, out.norm, out.penalty, out.critic_loss])
    base, gp = nn.WassersteinGan(g, c, x, z, a, 0), nn.WassersteinGan(g, c, x, z, a, 10)
    assert base.critic_loss.item() == -base.critic_gap.item()
    assert base.generator_loss.item() == gp.generator_loss.item()
    assert base.penalty.item() == gp.penalty.item()
    assert gp.critic_loss.item() == -gp.critic_gap.item()+10*gp.penalty.item()


@pytest.mark.parametrize('gp', [False, True])
def test_three_critic_steps_then_fresh_G_use_separate_optimizers_and_explicit_clips(gp):
    g, c, x, z, a = inputs()
    nn.WassersteinGan(g, c, x, z, a)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam if gp else nn.OptimizerKind.RMSProp
    config.beta1, config.beta2, config.rho = 0, .9, .9
    if not gp: nn.ClipWassersteinCritic(c)
    go, critic = nn.Optimizer(g, .0001 if gp else .00005, config), nn.Optimizer(c, .0001 if gp else .00005, config)
    for step in range(3):
        before = list(g.flatten())
        z = nn.GaussianRejectionPrior(2, 2, 5601+step, 2, nn.DType.Float64).z
        a = nn.MidpointUniform(2, 1, 6601+step, nn.DType.Float64)
        out = nn.WassersteinGan(g, c, x, z, a, 10 if gp else 0)
        critic.step(nn.grad(out.critic_loss, list(c.parameters())))
        if not gp:
            nn.ClipWassersteinCritic(c)
            assert max(abs(v) for v in c.flatten()) <= .01
        assert list(g.flatten()) == before
    before = list(c.flatten())
    z = nn.GaussianRejectionPrior(2, 2, 7601, 2, nn.DType.Float64).z
    a = nn.MidpointUniform(2, 1, 8601, nn.DType.Float64)
    out = nn.WassersteinGan(g, c, x, z, a, 10 if gp else 0)
    go.step(nn.grad(out.generator_loss, list(g.parameters())))
    assert list(c.flatten()) == before
    assert critic.iterations() == 3 and go.iterations() == 1


def test_owned_records_outlive_parents_and_all_invalid_inputs_precede_leaves():
    g, c, x, z, a = inputs()
    out = nn.WassersteinGan(g, c, x, z, a)
    loss, fake, interpolation = out.generator_loss, out.fake, out.interpolation
    del out, g, c
    gc.collect()
    assert tuple(fake.shape()) == (2, 2) and tuple(interpolation.shape()) == (2, 2)
    assert any(abs(v) > 1e-10 for v in nn.grad(loss, [z])[0].values())
    g, c, x, z, a = inputs()
    with pytest.raises(RuntimeError): nn.WassersteinGan(g, g, x, z, a)
    with pytest.raises(RuntimeError): nn.WassersteinGan(g, c, x, z, nn.full([2, 1], 2, nn.DType.Float64))
    with pytest.raises(RuntimeError): nn.WassersteinGan(g, c, x, z, a, -1)
    with pytest.raises(RuntimeError): nn.WassersteinGenerate(g, nn.full([2, 2], math.nan, nn.DType.Float64))
    assert g.size() == c.size() == 0
    with pytest.raises(RuntimeError): nn.ClipWassersteinCritic(c)
