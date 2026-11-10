"""Native affine code paths, detached class means and code-only decoding."""
import gc
import pytest
from pydatamunge import nn


def test_values_extrapolation_strides_and_independent_flatten():
    a = nn.from_vector_double([1, 2, 3, 4], [2, 2])
    b = nn.from_vector_double([5, 6, 7, 8], [2, 2])
    levels = nn.from_vector_double([0, .5, 1, 2], [4])
    path = nn.LatentLerp(a, b, levels)
    assert tuple(path.shape()) == (2, 4, 2)
    assert nn.equal_bits(nn.select(path, 1, 0), a)
    assert nn.equal_bits(nn.select(path, 1, 2), b)
    assert path.values()[6] == 9
    strided = nn.transpose(nn.transpose(a).contiguous())
    assert not strided.is_contiguous()
    assert nn.equal_bits(nn.LatentLerp(strided, b, levels), path)
    edited = nn.LatentDirection(a, b, nn.from_vector_double([-1, 0, 2], [3]))
    assert edited.values()[0] == -4
    assert nn.equal_bits(nn.select(edited, 1, 1), a)
    rows = nn.LatentFlatten(nn.transpose(path, [1, 0, 2]))
    assert tuple(rows.shape()) == (8, 2) and rows.is_contiguous()
    assert tuple(rows.values()) == tuple(nn.transpose(path, [1, 0, 2]).contiguous().values())
    with nn.no_grad():
        rows.fill_(7)
    assert path.values()[0] == 1


def test_all_affine_inputs_have_live_derivatives():
    t = nn.DType.Float64
    a = nn.full([2, 3], .4, t).requires_grad_()
    b = nn.full([2, 3], -.7, t).requires_grad_()
    levels = nn.from_vector_double([-.3, 0, .6, 1.4], [4]).requires_grad_()
    g = nn.grad(nn.sum(nn.LatentLerp(a, b, levels)), [a, b, levels])
    assert tuple(g[0].values()) == pytest.approx((2.3,)*6, abs=1e-14)
    assert tuple(g[1].values()) == pytest.approx((1.7,)*6, abs=1e-14)
    assert tuple(g[2].values()) == pytest.approx((-6.6,)*4, abs=1e-14)
    g = nn.grad(nn.sum(nn.LatentDirection(a, b, levels)), [a, b, levels])
    assert tuple(g[0].values()) == (4,)*6
    assert tuple(g[1].values()) == pytest.approx((1.7,)*6, abs=1e-14)
    assert tuple(g[2].values()) == pytest.approx((-4.2,)*4, abs=1e-14)


def test_centroids_detach_and_own_lifetime():
    code = nn.from_vector_double([1, -3, 2, 4, 4, 9], [3, 2]).requires_grad_()
    labels = nn.from_vector_int64([0, 1, 0], [3])
    means = nn.LatentClassMeans(code, labels, 2)
    assert tuple(means.values()) == (2.5, 3, 2, 4)
    assert not means.requires_grad()
    assert nn.equal_bits(nn.LatentClassMeans(nn.transpose(nn.transpose(code).contiguous()), labels, 2), means)
    del code, labels
    gc.collect()
    assert tuple(means.values()) == (2.5, 3, 2, 4)


def test_decoder_needs_only_codes_and_keeps_live_paths():
    store = nn.ParameterStore(1471)
    full = nn.UndercompleteAutoencoder(store, nn.full([2, 4], .3), 4, 2)
    before = tuple(store.flatten())
    prediction = nn.UndercompleteDecoder(store, full.code, 4, 2)
    assert nn.equal_bits(prediction, full.reconstruction)
    assert tuple(store.flatten()) == before
    decoder = nn.ParameterStore(1471)
    code = nn.full([2, 32], -.3).requires_grad_()
    output = nn.UndercompleteDecoder(decoder, code)
    assert decoder.size() == 2 and decoder.count() == 25872
    assert nn.sum(nn.abs(nn.grad(nn.sum(output), [code])[0])).item() > 0
    del store, full, decoder, code
    gc.collect()
    assert tuple(prediction.shape()) == (2, 4)


def test_invalid_inputs_missing_classes_and_decoder_preleaf_checks():
    a = nn.ones([2, 3])
    for levels in [nn.zeros([0]), nn.ones([2], nn.DType.Float64), nn.full([2], float('nan'))]:
        with pytest.raises(RuntimeError):
            nn.LatentLerp(a, a, levels)
    for labels in [nn.zeros([2]), nn.zeros([1], nn.DType.Int64), nn.zeros([2], nn.DType.Int64), nn.full([2], 2, nn.DType.Int64)]:
        with pytest.raises(RuntimeError):
            nn.LatentClassMeans(a, labels, 2)
    with pytest.raises(RuntimeError):
        nn.LatentFlatten(a)
    decoder = nn.ParameterStore(1471)
    for code in [nn.zeros([0, 32]), nn.zeros([2, 31]), nn.full([2, 32], float('inf'))]:
        with pytest.raises(RuntimeError):
            nn.UndercompleteDecoder(decoder, code)
    assert decoder.size() == 0
