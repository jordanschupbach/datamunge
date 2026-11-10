"""Native frozen SVD projection and deterministic fitting-neighbor diagnostics."""
import gc
import pytest
from pydatamunge import nn


def test_sample_rows_copy_strides_widen_and_keep_gradients():
    x = nn.transpose(nn.from_vector_double(list(range(12)), [2, 3, 2]), [1, 0, 2]).requires_grad_()
    rows = nn.DoubleSampleRows(x)
    assert tuple(rows.shape()) == (3, 4) and rows.dtype() == nn.DType.Float64
    assert tuple(rows.values()) == tuple(x.contiguous().values())
    assert tuple(nn.grad(nn.sum(rows), [x])[0].values()) == (1,)*12
    with nn.no_grad():
        rows.fill_(9)
    assert x.values()[0] == 0
    assert tuple(nn.DoubleSampleRows(nn.ones([3])).shape()) == (3, 1)


def test_pca_sample_variance_frozen_state_exports_and_query_paths():
    fit = nn.from_vector_double([-1, 0, 0, 0, 1, 0], [3, 2]).requires_grad_()
    pca = nn.SvdPca(fit, 1)
    assert tuple(pca.mean().values()) == (0, 0)
    assert pca.variances().item() == pytest.approx(1, abs=1e-14)
    assert not pca.scores().requires_grad() and not pca.components().requires_grad()
    query = nn.from_vector_double([2, 3], [1, 2]).requires_grad_()
    score = pca.Transform(query)
    recovered = pca.Reconstruct(score)
    assert tuple(recovered.values()) == pytest.approx((2, 0), abs=1e-14)
    assert nn.sum(nn.abs(nn.grad(nn.sum(nn.square(score)), [query])[0])).item() > 0
    before = tuple(pca.Transform(query).values())
    pca.mean().fill_(9); pca.components().fill_(9); pca.variances().fill_(9); pca.scores().fill_(9)
    with nn.no_grad():
        fit.fill_(9)
    assert tuple(pca.Transform(query).values()) == before
    kept = pca.components()
    del pca, fit
    gc.collect()
    assert tuple(kept.shape()) == (2, 1)


def test_prefix_inverse_constant_columns_and_no_variance_scaling():
    fit = nn.from_vector_double([-3, -1, -3, 1, 3, -1, 3, 1], [4, 2])
    pca = nn.SvdPca(fit, 2)
    assert tuple(pca.variances().values()) == pytest.approx((12, 4/3), abs=1e-12)
    score = pca.Transform(nn.from_vector_double([2, .5], [1, 2]))
    first = nn.slice(score, [0, 0], [1, 1])
    assert tuple(pca.Reconstruct(first).values()) == pytest.approx((2, 0), abs=1e-13)
    constant = nn.SvdPca(nn.full([3, 4], 7), 3)
    assert tuple(constant.variances().values()) == (0, 0, 0)
    assert tuple(constant.Reconstruct(constant.scores()).values()) == (7,)*12


def test_neighbor_ties_flattening_membership_and_posthoc_labels():
    fit = nn.from_vector_double([-1, 0, 1, 0, 0, 2, 4, 0], [4, 2])
    query = nn.zeros([1, 2]).requires_grad_()
    rows = nn.NearestFitRows(fit, query, 4)
    assert tuple(rows.values()) == (0, 1, 2, 3) and rows.dtype() == nn.DType.Int32
    assert not rows.requires_grad()
    assert nn.equal_bits(rows, nn.NearestFitRows(fit.reshape([4, 1, 2, 1]), query.reshape([1, 2, 1]), 4))
    a = nn.from_vector_int64([0, 1, 1, 1], [2, 2])
    b = nn.from_vector_int64([1, 2, 1, 1], [2, 2])
    assert nn.NeighborOverlap(a, b) == .75
    first = nn.from_vector_int64([0, 1, 1, 0], [2, 2])
    fit_labels = nn.from_vector_int64([4, 8], [2])
    assert nn.NeighborLabelAccuracy(first, fit_labels, nn.from_vector_int64([4, 9], [2])) == .5
    del fit, query
    gc.collect()
    assert tuple(rows.values()) == (0, 1, 2, 3)


def test_invalid_inputs_counts_neighbors_and_labels_are_rejected():
    for bad in [nn.zeros([0, 2]), nn.ones([3, 2], nn.DType.Int32), nn.full([3, 2], float('nan'))]:
        with pytest.raises(RuntimeError):
            nn.DoubleSampleRows(bad)
    for fit, count in [(nn.ones([1, 3]), 1), (nn.ones([3, 2]), 0), (nn.ones([2, 3]), 3)]:
        with pytest.raises(RuntimeError):
            nn.SvdPca(fit, count)
    pca = nn.SvdPca(nn.ones([3, 2]), 1)
    with pytest.raises(RuntimeError):
        pca.Transform(nn.ones([2, 3]))
    with pytest.raises(RuntimeError):
        pca.Reconstruct(nn.ones([2, 2]))
    with pytest.raises(RuntimeError):
        nn.NearestFitRows(nn.ones([2, 3]), nn.ones([1, 3]), 3)
    with pytest.raises(RuntimeError):
        nn.NeighborOverlap(nn.ones([2, 2]), nn.ones([2, 2]))
    with pytest.raises(RuntimeError):
        nn.NeighborLabelAccuracy(nn.full([2, 2], 2, nn.DType.Int32), nn.zeros([2], nn.DType.Int64), nn.zeros([2], nn.DType.Int64))
