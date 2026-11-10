"""Native signed IMQ discrepancy, explicit InfoVAE endpoints and host-owned updates."""
import gc
import math
import pytest
from pydatamunge import nn


def test_signed_unequal_mmd_owns_all_seven_fields_and_both_sample_paths():
    t = nn.DType.Float64
    x = nn.from_vector_double([0, 1], [2, 1]).requires_grad_()
    y = nn.from_vector_double([0, 2, 3], [3, 1]).requires_grad_()
    out = nn.ImqMmd(x, y)
    fields = {name: getattr(out, name) for name in ['xx', 'yy', 'xy', 'within_x', 'within_y', 'cross', 'value']}
    del out
    gc.collect()
    assert tuple(fields['xy'].shape()) == (2, 3)
    assert fields['within_x'].item() == pytest.approx(127/170)
    assert fields['value'].item() == pytest.approx(fields['within_x'].item() + fields['within_y'].item() - 2*fields['cross'].item())
    assert nn.ImqMmd(x, x).value.item() == pytest.approx(-43/170)
    kernel = nn.ImqKernel(nn.zeros([2, 1], t), nn.ones([3, 1], t))
    assert list(kernel.values()) == pytest.approx([127/170]*6)
    gradients = nn.grad(fields['value'], [x, y])
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients)


def test_objective_coefficients_preserve_signed_estimates_and_all_three_paths():
    t = nn.DType.Float64
    nll = nn.from_vector_double([4, 6], [2]).requires_grad_()
    kl = nn.from_vector_double([2, 4], [2]).requires_grad_()
    discrepancy = nn.full([], .25, t).requires_grad_()
    loss = nn.InfoVaeObjective(nll, kl, discrepancy)
    assert loss.item() == 8.875
    assert nn.InfoVaeObjective(nll, kl, discrepancy, 1).item() == 7.5
    assert nn.InfoVaeObjective(nll, kl, discrepancy, 0, 1).item() == 8
    assert nn.InfoVaeObjective(nll, kl, nn.neg(discrepancy)).item() == 4.125
    gradients = nn.grad(loss, [nll, kl, discrepancy])
    assert list(gradients[0].values()) == [.5, .5]
    assert list(gradients[1].values()) == [.25, .25]
    assert gradients[2].item() == 9.5


def test_all_sixteen_fields_survive_collection_and_all_live_float64_paths():
    t = nn.DType.Float64
    for alpha in [.5, 1]:
        p = nn.ParameterStore(3901, t)
        image = nn.full([2, 6], .3, t).requires_grad_()
        noise = nn.full([2, 2], .2, t).requires_grad_()
        prior = nn.full([3, 2], -.4, t).requires_grad_()
        out = nn.InfoVariationalAutoencoder(p, image, noise, prior, alpha, 10, 6, 2, 3)
        names = ['mu', 'log_variance', 'z', 'logits', 'probability', 'nll', 'kl', 'negative_elbo',
                 'mmd_xx', 'mmd_yy', 'mmd_xy', 'mmd_within_x', 'mmd_within_y', 'mmd_cross', 'mmd', 'loss']
        fields = {name: getattr(out, name) for name in names}
        del out
        gc.collect()
        assert p.size() == 10 and p.count() == 70
        assert tuple(fields['mmd_xy'].shape()) == (2, 3)
        assert tuple(fields['probability'].shape()) == (2, 6) and fields['loss'].dtype() == t
        before = tuple(p.flatten())
        assert list(nn.sigmoid(nn.InfoVaeDecode(p, fields['z'], 6, 2, 3)).values()) == list(fields['probability'].values())
        assert tuple(p.flatten()) == before
        gradients = nn.grad(fields['loss'], list(p.parameters()) + [image, noise, prior])
        assert len(gradients) == 13 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
        assert all(any(abs(v) > 1e-12 for v in g.values()) for g in gradients[-3:])


def test_default_census_prior_independence_and_visible_adam():
    p = nn.ParameterStore(3901)
    image, noise, prior = nn.full([2, 784], .3), nn.ones([2, 4]), nn.zeros([3, 4])
    out = nn.InfoVariationalAutoencoder(p, image, noise, prior)
    assert p.size() == 10 and p.count() == 26104
    other = nn.InfoVariationalAutoencoder(p, image, noise, nn.ones([4, 4]))
    assert list(other.probability.values()) == list(out.probability.values())
    assert other.mmd.item() != out.mmd.item() and other.loss.item() != out.loss.item()
    before = tuple(p.flatten())
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    optimizer.step(nn.grad(out.loss, list(p.parameters())))
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before
    decoder = nn.ParameterStore(1)
    assert tuple(nn.InfoVaeDecode(decoder, nn.zeros([1, 4])).shape()) == (1, 784)
    assert decoder.size() == 4 and decoder.count() == 13408


def test_invalid_controls_counts_dtype_and_pixels_fail_before_creation():
    p = nn.ParameterStore(1)
    x, e, prior = nn.full([2, 6], .3), nn.ones([2, 2]), nn.zeros([3, 2])
    for alpha, weight in [(-.1, 10), (1.1, 10), (.5, .9), (.5, float('inf')), (1e-100, 10)]:
        with pytest.raises(RuntimeError):
            nn.InfoVariationalAutoencoder(p, x, e, prior, alpha, weight, 6, 2, 3)
    with pytest.raises(RuntimeError):
        nn.InfoVariationalAutoencoder(p, nn.full([2, 6], 1.1), e, prior, .5, 10, 6, 2, 3)
    with pytest.raises(RuntimeError):
        nn.InfoVariationalAutoencoder(p, x, e, nn.zeros([1, 2]), .5, 10, 6, 2, 3)
    assert p.size() == 0
    with pytest.raises(RuntimeError):
        nn.ImqMmd(nn.zeros([1, 2]), prior)
    with pytest.raises(RuntimeError):
        nn.ImqKernel(e, nn.ones([2, 2], nn.DType.Float64))
    with pytest.raises(RuntimeError):
        nn.InfoVaeObjective(nn.ones([2]), nn.ones([3]), nn.scalar(0))
