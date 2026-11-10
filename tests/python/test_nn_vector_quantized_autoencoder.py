"""Native VQ routes, owned nested fields and explicit fitting-prior uniforms."""
import gc
import math
import pytest
from pydatamunge import nn


def _ids(values, shape):
    return nn.cast(nn.from_vector_int64(values, shape), nn.DType.Int32)


def test_quantizer_tie_surrogate_dictionary_and_commitment_routes():
    encoded = nn.from_vector_double([-.5, 0, .9, .2], [2, 2]).requires_grad_()
    book = nn.from_vector_double([-1, 0, 0, 0, 1, 0], [3, 2]).requires_grad_()
    out = nn.VectorQuantize(encoded, book, 3, 2)
    assert tuple(out.indices.values()) == (0, 2)
    assert nn.equal_bits(out.quantized, out.straight_through)
    assert out.assignment.dtype() == nn.DType.Float64
    zero_encoded = nn.mul(nn.sum(encoded), 0)
    zero_book = nn.mul(nn.sum(book), 0)
    reconstruction = nn.grad(nn.add(nn.mul(nn.sum(nn.square(out.straight_through)), .5), zero_book), [encoded, book])
    dictionary = nn.grad(nn.add(out.codebook_loss, zero_encoded), [encoded, book])
    commitment = nn.grad(nn.add(nn.mul(out.commitment_loss, .25), nn.mul(nn.sum(book), 0)), [encoded, book])
    assert tuple(reconstruction[0].values()) == tuple(out.quantized.values())
    assert tuple(reconstruction[1].values()) == (0,)*6
    assert tuple(dictionary[0].values()) == (0,)*4
    assert tuple(dictionary[1].values()) == pytest.approx((-.25, 0, 0, 0, .05, -.1), abs=1e-14)
    assert tuple(commitment[0].values()) == pytest.approx((.0625, 0, -.0125, .025), abs=1e-14)
    assert tuple(commitment[1].values()) == (0,)*6
    override = _ids([2, 1], [1, 2])
    changed = nn.VectorQuantize(encoded, book, 3, 2, override)
    assert tuple(changed.indices.values()) == (2, 1)
    assert nn.equal_bits(changed.quantized, nn.VqLookup(override, book, 3))


def test_nested_quantizer_ownership_general_spatial_model_and_caller_adam():
    p = nn.ParameterStore(2707, nn.DType.Float64)
    image = nn.full([2, 8, 12, 1], .3, nn.DType.Float64).requires_grad_()
    out = nn.VectorQuantizedAutoencoder(p, image, 3, 2, 3, .5)
    assert p.size() == 9
    quantizer, probability, loss = out.vq, out.probability, out.loss
    field_names = ('indices', 'assignment', 'quantized', 'straight_through', 'codebook_loss', 'commitment_loss')
    retained = {name: getattr(quantizer, name) for name in field_names}
    snapshots = {name: tuple(value.values()) for name, value in retained.items()}
    expected = tuple(probability.values())
    del out, quantizer
    gc.collect()
    for name, value in retained.items():
        assert tuple(value.values()) == snapshots[name]
    ids = retained['indices'].reshape([2, 6])
    decoded = nn.sigmoid(nn.VqDecoder(p, nn.VqLookup(ids, p.at('codebook'), 3), 8, 12, 2, 3))
    assert tuple(decoded.values()) == expected
    derivatives = nn.grad(loss, list(p.parameters())+[image])
    assert len(derivatives) == 10
    assert all(math.isfinite(v) for g in derivatives for v in g.values())
    config = nn.OptimizerConfig();config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten());optimizer.step(derivatives[:9])
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_default_census_code_only_decoder_and_rejected_geometry():
    p = nn.ParameterStore(2707)
    out = nn.VectorQuantizedAutoencoder(p, nn.full([2, 28, 28, 1], .3))
    assert p.size() == 9 and p.count() == 805
    assert tuple(out.encoded.shape()) == (98, 4)
    ids = out.vq.indices.reshape([2, 49])
    assert nn.equal_bits(out.probability, nn.sigmoid(nn.VqDecoder(p, nn.VqLookup(ids, out.book))))
    empty = nn.ParameterStore(1)
    for entries, dimensions, channels, beta in [(0, 4, 8, .25), (16, 0, 8, .25), (16, 4, 0, .25), (16, 4, 8, -1)]:
        with pytest.raises(RuntimeError):
            nn.VectorQuantizedAutoencoder(empty, nn.zeros([1, 28, 28, 1]), entries, dimensions, channels, beta)
    assert empty.size() == 0
    with pytest.raises(RuntimeError):
        nn.VqLookup(_ids([-1], [1, 1]), out.book)


def test_prior_owns_fitting_state_explicit_cdf_boundaries_and_usage():
    fit = _ids([0, 1, 1, 0], [2, 2])
    prior = nn.VqMarkovPrior(fit, 2)
    del fit
    gc.collect()
    assert tuple(prior.counts().values()) == (0, 1, 1, 0, 1, 1)
    assert tuple(prior.probabilities().values()) == (.25, .75, .75, .25, .5, .5)
    exported = prior.counts();exported.fill_(99)
    assert tuple(prior.counts().values()) == (0, 1, 1, 0, 1, 1)
    u = nn.from_vector_double([.5, .75, .25, .5, .1, .75], [2, 3])
    ids = prior.Sample(u)
    assert tuple(ids.values()) == (1, 1, 0, 1, 0, 1)
    assert prior.Nll(_ids([0, 1, 1, 0], [2, 2])) == pytest.approx(-math.log(.5)-math.log(.75), abs=1e-14)
    bank = nn.MidpointUniformNoise([20, 49], 2729)
    assert nn.equal_bits(prior.Sample(bank), prior.Sample(bank))
    usage = nn.VqUsage(_ids([0, 1, 0, 1], [2, 2]), 4)
    counts = usage.counts;del usage;gc.collect()
    assert tuple(counts.values()) == (2, 2, 0, 0)
    assert nn.VqUsage(_ids([0, 1], [1, 2]), 4).perplexity == 2
    for uniform in [nn.zeros([1, 2]), nn.full([1, 2], 1, nn.DType.Float64)]:
        with pytest.raises(RuntimeError):
            prior.Sample(uniform)
    with pytest.raises(RuntimeError):
        nn.VqMarkovPrior(nn.zeros([1, 2], nn.DType.Int32), 2, 0)
