"""Native conditional affine flow, density signs, output ownership and explicit host fitting."""
import gc
import math
import pytest
from pydatamunge import nn


def test_hand_coupling_inverse_ordering_and_live_context():
    t = nn.DType.Float64
    p = nn.ParameterStore(3049, t)
    z = nn.from_vector_double([.4, -.2], [1, 2]).requires_grad_()
    context = nn.ones([1, 1], t).requires_grad_()
    nn.ConditionalAffinePosteriorFlow(p, 'f', z, context, False, False, 2, 2)
    p.set('f/heads/W', nn.zeros([2, 2], t))
    p.set('f/heads/b', nn.from_vector_double([math.atanh(.4), .3], [2]))
    for swap in [False, True]:
        out = nn.ConditionalAffinePosteriorFlow(p, 'f', z, context, swap, False, 2, 2)
        back = nn.ConditionalAffinePosteriorFlow(p, 'f', out.value, context, swap, True, 2, 2)
        assert list(back.value.values()) == pytest.approx(list(z.values()), abs=1e-14)
        assert out.log_determinant.item() == pytest.approx(.2)
        assert back.log_determinant.item() == pytest.approx(-.2)
    q = nn.ParameterStore(3049, t)
    out = nn.ConditionalAffinePosteriorFlow(q, 'g', z, context, False, False, 2, 2)
    value, ld = out.value, out.log_determinant
    del out
    gc.collect()
    gradients = nn.grad(nn.add(nn.sum(value), nn.sum(ld)), [z, context])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_all_named_fields_own_lifetime_and_host_owns_optimizer():
    p = nn.ParameterStore(3001)
    x, epsilon = nn.full([2, 3], .3), nn.full([2, 2], .7)
    out = nn.FlowVariationalAutoencoder(p, x, epsilon, 3, 2, 2, 2)
    assert p.size() == 18 and p.count() == 63
    arrays = dict(out.Named())
    assert len(arrays) == 18 and list(arrays)[0] == 'context' and list(arrays)[-1] == 'inverse_logdet'
    loss, z, probability = out.loss, out.z, out.probability
    expected = list(probability.values())
    del out
    gc.collect()
    assert list(nn.sigmoid(nn.GaussianVaeDecode(p, z, 3, 2, 2)).values()) == expected
    assert list(nn.sub(arrays['logq0'], arrays['logdet']).values()) == list(arrays['logq'].values())
    assert list(nn.sub(arrays['logq'], arrays['logp']).values()) == list(arrays['logratio'].values())
    assert list(arrays['inverse_z0'].values()) == pytest.approx(list(arrays['z0'].values()), abs=1e-6)
    gradients = nn.grad(loss, list(p.parameters()))
    assert len(gradients) == 18 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_default_census_float64_and_direct_prior_decoder():
    p = nn.ParameterStore(3001)
    out = nn.FlowVariationalAutoencoder(p, nn.zeros([1, 784]), nn.zeros([1, 4]))
    assert p.count() == 52672 and p.size() == 18
    assert list(out.z0.values()) == list(out.mu.values())
    decoder = nn.ParameterStore(1)
    nn.GaussianVaeDecode(decoder, nn.zeros([20, 4]))
    assert decoder.size() == 4 and decoder.count() == 26032
    q = nn.ParameterStore(3001, nn.DType.Float64)
    f = nn.FlowVariationalAutoencoder(q, nn.full([2, 3], .3, nn.DType.Float64), nn.ones([2, 2], nn.DType.Float64), 3, 2, 2, 2)
    assert f.loss.dtype() == nn.DType.Float64 and math.isfinite(f.loss.item())


def test_invalid_inputs_leave_store_empty():
    p = nn.ParameterStore(1)
    x, e = nn.zeros([2, 3]), nn.zeros([2, 2])
    for widths in [(3, 3, 2, 2), (3, 2, 2, 0), (3, 2, 0, 2)]:
        with pytest.raises(RuntimeError):
            nn.FlowVariationalAutoencoder(p, x, e, *widths)
    with pytest.raises(RuntimeError):
        nn.FlowVariationalAutoencoder(p, nn.full([2, 3], 1.1), e, 3, 2, 2, 2)
    with pytest.raises(RuntimeError):
        nn.ConditionalAffinePosteriorFlow(p, '', e, x, False, False, 2, 2)
    assert p.size() == 0
