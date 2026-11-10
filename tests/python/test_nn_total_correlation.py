"""Total correlation, nested result ownership and explicit alternating host updates."""
import gc
import math
import pytest
from pydatamunge import nn


def test_exact_mixture_signed_hand_values_and_three_live_inputs():
    dtype = nn.DType.Float64
    z = nn.from_vector_double([1., 1., 1., -1.], [2, 2]).requires_grad_()
    mu = nn.from_vector_double([-1., -1., 1., 1.], [2, 2]).requires_grad_()
    lv = nn.zeros([2, 2], dtype).requires_grad_()
    out = nn.EmpiricalTotalCorrelation(z, mu, lv)
    v = math.exp(-2)
    expected = [math.log(2*(1+v*v)/(1+v)**2), math.log(4*v/(1+v)**2)]
    assert list(out.per_case.values()) == pytest.approx(expected, abs=1e-14)
    mean, joint = out.mean, out.log_joint
    del out
    gc.collect()
    assert mean.item() < 0 and tuple(joint.shape()) == (2,)
    gradients = nn.grad(mean, [z, mu, lv])
    assert len(gradients) == 3
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_nested_result_owns_gaussian_fields_and_two_phase_updates():
    p, d = nn.ParameterStore(2237), nn.ParameterStore(2309)
    x, epsilon = nn.full([2, 3], .3), nn.full([2, 2], .7)
    out = nn.FactorVariationalAutoencoder(p, d, x, epsilon, 3, 3, 2, 2, 2)
    assert p.count() == 35 and d.count() == 9
    net, loss, tc = out.net, out.loss, out.tc
    del out
    gc.collect()
    z, probability = net.z, net.probability
    del net
    gc.collect()
    assert tuple(probability.shape()) == (2, 3) and math.isfinite(tc.item())
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer, disc_optimizer = nn.Optimizer(p, .003, config), nn.Optimizer(d, .003, config)
    disc_before = tuple(d.flatten())
    optimizer.step(nn.grad(loss, list(p.parameters())))
    assert tuple(d.flatten()) == disc_before
    # Fresh posterior samples from the updated VAE, then an explicitly detached D phase.
    fresh = nn.FactorVariationalAutoencoder(p, d, x, epsilon, 3, 3, 2, 2, 2)
    joint = fresh.net.z.detach()
    product = nn.PermuteLatentColumns(joint, 2311)
    assert not product.requires_grad()
    assert tuple(product.values()) == tuple(nn.PermuteLatentColumns(joint, 2311).values())
    before = tuple(p.flatten())
    disc_loss = nn.FactorDiscriminatorLoss(d, joint, product, 2, 2)
    disc_optimizer.step(nn.grad(disc_loss, list(d.parameters())))
    assert tuple(p.flatten()) == before
    assert optimizer.iterations() == disc_optimizer.iterations() == 1
    assert tuple(z.shape()) == (2, 2)


def test_full_reference_gradient_and_nested_beta_result_survive_collection():
    p = nn.ParameterStore(2237)
    refs = nn.from_vector([.2, .3, .7, .5, .4, .1, .8, .2, .6], [3, 3]).requires_grad_()
    out = nn.BetaTcVariationalAutoencoder(p, nn.full([2, 3], .3), nn.ones([2, 2]), refs, 3, 3, 2, 2)
    net, loss = out.net, out.loss
    del out
    gc.collect()
    gradients = nn.grad(loss, list(p.parameters()) + [refs])
    assert len(gradients) == 11
    assert any(abs(v) > 1e-12 for v in gradients[-1].values())
    assert tuple(net.mu.shape()) == (2, 2)
    zero = nn.BetaTcVariationalAutoencoder(p, nn.full([2, 3], .3), nn.ones([2, 2]), refs, 0, 3, 2, 2)
    assert zero.loss.item() == zero.net.unweighted_loss.item()


def test_invalid_reference_weight_and_alias_fail_before_creating_parameters():
    p = nn.ParameterStore(1)
    x, e = nn.zeros([2, 3]), nn.zeros([2, 2])
    for weight in [-1, float('nan'), float('inf')]:
        with pytest.raises(RuntimeError):
            nn.BetaTcVariationalAutoencoder(p, x, e, nn.zeros([3, 3]), weight, 3, 2, 2)
    with pytest.raises(RuntimeError):
        nn.BetaTcVariationalAutoencoder(p, x, e, nn.zeros([0, 3]), 3, 3, 2, 2)
    with pytest.raises(RuntimeError):
        nn.FactorVariationalAutoencoder(p, p, x, e, 3, 3, 2, 2, 2)
    assert p.size() == 0
