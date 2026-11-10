"""Native two-scale VQ ownership, hierarchy paths and explicit conditional prior."""
import gc
import math
import pytest
from pydatamunge import nn


def ids(values, shape):
    return nn.cast(nn.from_vector_int64(values, shape), nn.DType.Int32)


def test_default_census_and_two_grid_only_decode():
    p = nn.ParameterStore(2801)
    out = nn.VectorQuantizedAutoencoder2(p, nn.full([2, 28, 28, 1], .3))
    assert p.size() == 14 and p.count() == 1157
    assert tuple(out.encoded_top.shape()) == (98, 4)
    assert tuple(out.encoded_bottom.shape()) == (392, 4)
    assert tuple(out.top_feature.shape()) == (2, 14, 14, 4)
    assert nn.equal_bits(out.top.quantized, out.top.straight_through)
    assert nn.equal_bits(out.bottom.quantized, out.bottom.straight_through)
    top = nn.VqLookup(out.top.indices.reshape([2, 49]), out.top_book)
    bottom = nn.VqLookup(out.bottom.indices.reshape([2, 196]), out.bottom_book)
    probability = nn.sigmoid(nn.Vq2Decoder(p, bottom, nn.Vq2TopDecoder(p, top)))
    assert nn.equal_bits(probability, out.probability)


def test_eleven_fields_nested_lifetimes_general_float64_and_caller_adam():
    t = nn.DType.Float64
    p = nn.ParameterStore(2801, t)
    x = nn.full([2, 8, 12, 1], .3, t).requires_grad_()
    out = nn.VectorQuantizedAutoencoder2(p, x, entries=3, dimensions=2, channels=3)
    names = ('encoded_top', 'encoded_bottom', 'top_book', 'bottom_book', 'top_feature',
             'probability', 'nll', 'reconstruction_loss', 'loss')
    fields = {name: getattr(out, name) for name in names}
    top, bottom = out.top, out.bottom
    del out
    gc.collect()
    top_ids, bottom_ids = top.indices, bottom.indices
    top_value, bottom_value = top.quantized, bottom.quantized
    del top, bottom
    gc.collect()
    assert tuple(top_value.shape()) == (12, 2) and tuple(bottom_value.shape()) == (48, 2)
    assert all(math.isfinite(v) for value in fields.values() for v in value.values())
    feature = nn.Vq2TopDecoder(p, nn.VqLookup(top_ids.reshape([2, 6]), fields['top_book'], 3), 8, 12, 2)
    decoded = nn.sigmoid(nn.Vq2Decoder(p, nn.VqLookup(bottom_ids.reshape([2, 24]), fields['bottom_book'], 3), feature, 3))
    assert nn.equal_bits(decoded, fields['probability'])
    derivatives = nn.grad(fields['loss'], list(p.parameters())+[x])
    assert len(derivatives) == 15 and all(math.isfinite(v) for g in derivatives for v in g.values())
    config = nn.OptimizerConfig();config.kind = nn.OptimizerKind.Adam
    adam = nn.Optimizer(p, .003, config);before = tuple(p.flatten());adam.step(derivatives[:14])
    assert adam.iterations() == 1 and tuple(p.flatten()) != before


def test_fine_to_coarse_surrogate_and_both_codebook_stops():
    t = nn.DType.Float64;p = nn.ParameterStore(2801, t)
    image = nn.full([2, 8, 12, 1], .3, t)
    nn.VectorQuantizedAutoencoder2(p, image, entries=3, dimensions=2, channels=3)
    p.set('top_codebook', nn.from_vector_double([.1, .2, .5, .6, -.2, .3], [3, 2]))
    p.set('top_decoder/W', nn.full(list(p.at('top_decoder/W').shape()), .1, t))
    p.set('top_decoder/b', nn.full([2], .1, t))
    p.set('bottom_projection/W', nn.full(list(p.at('bottom_projection/W').shape()), .2, t))
    out = nn.VectorQuantizedAutoencoder2(p, image, entries=3, dimensions=2, channels=3)
    coupling = nn.grad(nn.mean(nn.square(out.encoded_bottom)), [out.encoded_top])
    assert any(abs(v) > 1e-10 for v in coupling[0].values())
    out = nn.VectorQuantizedAutoencoder2(p, image, entries=3, dimensions=2, channels=3)
    probe = nn.add(out.reconstruction_loss, nn.mul(nn.add(nn.sum(out.top_book), nn.sum(out.bottom_book)), 0))
    books = nn.grad(probe, [out.top_book, out.bottom_book])
    assert all(v == 0 for g in books for v in g.values())
    zero = nn.VectorQuantizedAutoencoder2(p, image, nn.zeros([2, 6], nn.DType.Int32), 3, 2, 3)
    one = nn.VectorQuantizedAutoencoder2(p, image, nn.ones([2, 6], nn.DType.Int32), 3, 2, 3)
    assert tuple(zero.encoded_bottom.values()) != tuple(one.encoded_bottom.values())
    assert tuple(one.top.indices.values()) == (1,)*12
    assert tuple(nn.grad(nn.sum(one.top.straight_through), [one.encoded_top])[0].values()) == (1,)*24


def test_invalid_overrides_and_geometry_precede_parameter_creation():
    p = nn.ParameterStore(2801);image = nn.zeros([1, 28, 28, 1])
    for top in [nn.zeros([1, 49]), nn.zeros([1, 48], nn.DType.Int32), nn.full([1, 49], 16, nn.DType.Int32)]:
        with pytest.raises(RuntimeError):
            nn.VectorQuantizedAutoencoder2(p, image, top)
    with pytest.raises(RuntimeError):
        nn.VectorQuantizedAutoencoder2(p, image, entries=0)
    with pytest.raises(RuntimeError):
        nn.VectorQuantizedAutoencoder2(p, nn.zeros([1, 27, 28, 1]))
    assert p.size() == 0


def test_conditional_prior_parents_bos_explicit_cdf_and_ownership():
    top, bottom = ids([0, 1], [1, 2]), ids([0, 0, 1, 1, 0, 0, 1, 1], [1, 8])
    prior = nn.Vq2ConditionalPrior(top, bottom, 1, 2, 2)
    assert tuple(prior.counts().values()) == (2, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0)
    expected = tuple(prior.probabilities().values())
    exported = prior.probabilities();exported.fill_(0)
    assert tuple(prior.probabilities().values()) == expected
    assert nn.equal_bits(prior.Sample(top, nn.full([1, 8], .2, nn.DType.Float64)), bottom)
    assert prior.Nll(top, bottom) == pytest.approx(-2*math.log(.75)-6*math.log(2.5/3), abs=1e-14)
    del top, bottom
    gc.collect()
    assert tuple(prior.counts().shape()) == (2, 3, 2)
    small_top, small_bottom = nn.zeros([1, 1], nn.DType.Int32), nn.zeros([1, 4], nn.DType.Int32)
    small = nn.Vq2ConditionalPrior(small_top, small_bottom, 1, 1, 2)
    assert tuple(small.Sample(small_top, nn.full([1, 4], .75, nn.DType.Float64)).values()) == (1,)*4
    bank = nn.MidpointUniformNoise([1, 4], 2833)
    assert nn.equal_bits(small.Sample(small_top, bank), small.Sample(small_top, bank))
    for uniform in [nn.zeros([1, 4]), nn.full([1, 4], 1, nn.DType.Float64)]:
        with pytest.raises(RuntimeError):
            small.Sample(small_top, uniform)
    with pytest.raises(RuntimeError):
        nn.Vq2ConditionalPrior(small_top, small_bottom, 1, 1, 2, 0)
