"""Native deterministic WAE, signed objective, nested ownership and host-owned Adam."""
import gc
import math
import pytest
from pydatamunge import nn


def test_pixel_sums_signed_objective_and_both_paths():
    t = nn.DType.Float64
    x = nn.full([2, 4], .2, t).requires_grad_()
    y = nn.full([2, 4], .7, t).requires_grad_()
    cost = nn.WaeSquaredPixelCost(x, y)
    assert tuple(cost.values()) == pytest.approx([1, 1])
    gradients = nn.grad(nn.mean(cost), [x, y])
    assert tuple(gradients[0].values()) == pytest.approx([-.5]*8)
    assert tuple(gradients[1].values()) == pytest.approx([.5]*8)
    c = nn.from_vector_double([2, 4], [2]).requires_grad_()
    mmd = nn.full([], -.1, t).requires_grad_()
    loss = nn.WaeObjective(c, mmd)
    assert loss.item() == 2
    gradients = nn.grad(loss, [c, mmd])
    assert tuple(gradients[0].values()) == (.5, .5)
    assert gradients[1].item() == 10
    assert nn.grad(nn.WaeObjective(c, mmd, 0), [mmd])[0].item() == 0
    assert nn.WaeObjective(nn.zeros([2]), nn.full([], -.1)).item() < 0


def test_default_census_decoder_only_restore_and_explicit_adam():
    p = nn.ParameterStore(3701)
    out = nn.WassersteinAutoencoder(p, nn.full([2, 784], .3), nn.zeros([3, 4]))
    assert p.size() == 8 and p.count() == 26036
    assert tuple(out.code.shape()) == (2, 4)
    assert tuple(out.mmd.xy.shape()) == (2, 3)
    assert nn.equal_bits(out.probability, nn.sigmoid(nn.WaeDecode(p, out.code)))
    decoder = nn.ParameterStore(1)
    nn.WaeDecode(decoder, nn.zeros([20, 4]))
    assert decoder.size() == 4 and decoder.count() == 13408
    for name in ['decoder/W', 'decoder/b', 'output/W', 'output/b']:
        decoder.set(name, p.at(name))
    assert nn.equal_bits(out.probability, nn.sigmoid(nn.WaeDecode(decoder, out.code)))
    before = tuple(p.flatten())
    config = nn.OptimizerConfig();config.kind = nn.OptimizerKind.Adam
    adam = nn.Optimizer(p, .003, config)
    adam.step(nn.grad(out.loss, list(p.parameters())))
    assert adam.iterations() == 1 and tuple(p.flatten()) != before


def test_all_fields_nested_gc_float64_strides_and_live_inputs():
    t = nn.DType.Float64;p = nn.ParameterStore(3701, t)
    x = nn.transpose(nn.full([6, 2], .3, t)).requires_grad_()
    prior = nn.transpose(nn.full([2, 3], -.2, t)).requires_grad_()
    assert not x.is_contiguous() and not prior.is_contiguous()
    out = nn.WassersteinAutoencoder(p, x, prior, 10, 6, 2, 3)
    names = ['code', 'logits', 'probability', 'squared_cost', 'reconstruction_loss', 'loss']
    fields = {name: getattr(out, name) for name in names};nested = out.mmd
    del out
    gc.collect()
    kernel_names = ['xx', 'yy', 'xy', 'within_x', 'within_y', 'cross', 'value']
    kernels = {name: getattr(nested, name) for name in kernel_names}
    del nested
    gc.collect()
    assert p.size() == 8 and p.count() == 62
    assert tuple(kernels['xy'].shape()) == (2, 3)
    assert all(math.isfinite(v) for value in [*fields.values(), *kernels.values()] for v in value.values())
    derivatives = nn.grad(fields['loss'], list(p.parameters())+[x, prior])
    assert len(derivatives) == 10
    assert all(math.isfinite(v) for g in derivatives for v in g.values())
    assert all(any(abs(v) > 1e-12 for v in g.values()) for g in derivatives[-2:])
    assert nn.equal_bits(fields['probability'], nn.sigmoid(nn.WaeDecode(p, fields['code'], 6, 2, 3)))


def test_prior_only_regularizer_unequal_counts_and_zero_weight():
    p = nn.ParameterStore(3701);x = nn.full([3, 784], .3)
    a = nn.WassersteinAutoencoder(p, x, nn.zeros([4, 4]))
    before = tuple(p.flatten())
    b = nn.WassersteinAutoencoder(p, x, nn.ones([5, 4]))
    assert nn.equal_bits(a.code, b.code) and nn.equal_bits(a.probability, b.probability)
    assert a.mmd.value.item() != b.mmd.value.item()
    small = nn.WassersteinAutoencoder(p, nn.take_rows(x, [0, 1]), nn.zeros([7, 4]))
    assert nn.equal_bits(small.probability, nn.take_rows(a.probability, [0, 1]))
    zero = nn.WassersteinAutoencoder(p, x, nn.zeros([4, 4]), 0)
    assert zero.loss.item() == zero.reconstruction_loss.item()
    assert tuple(p.flatten()) == before


def test_invalid_counts_controls_values_and_geometry_precede_leaves():
    p = nn.ParameterStore(1);x = nn.full([2, 6], .3);prior = nn.zeros([3, 2])
    for weight in [-1, float('inf'), float('nan'), 1e100, 1e-100]:
        with pytest.raises(RuntimeError):
            nn.WassersteinAutoencoder(p, x, prior, weight, 6, 2, 3)
    for image, ref in [(nn.zeros([1, 6]), prior), (x, nn.zeros([1, 2])),
                       (nn.full([2, 6], 1.1), prior), (x, nn.zeros([3, 2], nn.DType.Float64))]:
        with pytest.raises(RuntimeError):
            nn.WassersteinAutoencoder(p, image, ref, 0, 6, 2, 3)
    with pytest.raises(RuntimeError):
        nn.WaeDecode(p, nn.zeros([2, 3]), 6, 2, 3)
    assert p.size() == 0
    with pytest.raises(RuntimeError):
        nn.WaeSquaredPixelCost(nn.zeros([2, 3]), nn.full([2, 3], -.1))
    with pytest.raises(RuntimeError):
        nn.WaeObjective(nn.from_vector_double([-1, 2], [2]), nn.full([], .1, nn.DType.Float64))
