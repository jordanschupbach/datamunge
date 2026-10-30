import math

import pytest

from pydatamunge import datamunge as dm


def grid(n, lo=0.0, hi=1.0):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def test_tensor_bspline_evaluates_refines_and_fits():
    b = dm.BSplineBasis.open_uniform(1, 2)
    f = dm.TensorBSpline([b, b], [0.0, 1.0, 1.0, 2.0])  # x + y
    assert f.dimensions() == 2
    assert list(f.shape()) == [2, 2]
    assert math.isclose(f.evaluate([0.25, 0.5]), 0.75)
    assert list(f.evaluate_grid([[0.0, 1.0], [0.0, 1.0]])) == [0.0, 1.0, 1.0, 2.0]
    f.set_coefficient([1, 1], 4.0)
    assert f.coefficient([1, 1]) == 4.0

    c = dm.BSplineBasis.open_uniform(3, 6)
    curve = dm.TensorBSpline(c, [0.0, 1.0, -1.0, 2.0, 0.5, 1.0])
    refined = curve.dyadic_refine()
    assert refined.coefficient_count() > curve.coefficient_count()
    for x in grid(11):
        assert math.isclose(refined.evaluate(x), curve.evaluate(x), abs_tol=1e-12)

    r = dm.dyadic_refine(dm.BSplineBasis.open_uniform(1, 2))
    assert list(r.apply([0.0, 2.0])) == pytest.approx([0.0, 1.0, 2.0])

    pts = [[x, y] for x in grid(8) for y in grid(8)]
    vals = [math.sin(3 * p[0]) + p[1] for p in pts]
    q = dm.BSplineBasis.open_uniform(3, 6)
    plain = dm.fit_tensor_bspline([q, q], pts, vals)
    smooth = dm.fit_tensor_pspline([q, q], pts, vals, [1.0, 1.0], 2)
    zero = dm.fit_tensor_pspline([q, q], pts, vals, [0.0, 0.0])
    assert plain.evaluate([0.3, 0.3]) == pytest.approx(zero.evaluate([0.3, 0.3]), abs=1e-9)
    assert math.isfinite(smooth.evaluate([0.3, 0.3]))

    opts = dm.SplineSolveOptions()
    opts.method = dm.SplineSolver_ConjugateGradient
    cg = dm.fit_tensor_bspline([q, q], pts, vals, opts)
    assert cg.evaluate([0.7, 0.2]) == pytest.approx(plain.evaluate([0.7, 0.2]), abs=1e-7)


def test_hierarchical_overlay_spline():
    h = dm.HierarchicalBSpline([0.0], [1.0])
    const = dm.BSplineBasis.open_uniform(0, 1)
    h.set_spline(h.root(), dm.TensorBSpline(const, [1.0]))
    child = h.refine(h.root(), 1, dm.TensorBSpline(const, [2.0]))
    assert h.evaluate([0.25]) == 1.0
    assert h.evaluate([0.75]) == 3.0
    assert h.locate([0.9]) == child
    assert h.locate([2.0]) == -1
    assert list(h.cell_max(child)) == [1.0]

    lin = dm.BSplineBasis.open_uniform(1, 2)
    g = dm.HierarchicalBSpline([0.0], [1.0])
    g.fit_cell(g.root(), [lin], [[0.0], [1.0]], [1.0, 3.0])
    assert g.evaluate([0.5]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        g.fit_cell(g.root(), [lin], [[0.0], [1.0]], [1.0, 3.0])  # already has a patch
    with pytest.raises(IndexError):
        g.depth(42)


def test_truncated_hierarchical_spline():
    b = dm.BSplineBasis.open_uniform(2, 4)
    thb = dm.TruncatedHierarchicalBSpline(dm.TensorBSpline([b, b], [1.0] * 16))
    thb.refine_region([0.0, 0.0], [0.5, 0.5], 1)
    thb.refine_region([0.0, 0.0], [0.2, 0.2], 2)
    assert thb.level_count() == 3
    assert thb.evaluate([0.1, 0.1]) == pytest.approx(1.0)

    # Truncated basis is a partition of unity.
    active = [list(thb.active_indices(level)) for level in range(thb.level_count())]
    for p in ([0.05, 0.05], [0.3, 0.1], [0.8, 0.9]):
        total = sum(
            thb.truncated_basis(level, list(idx), p)
            for level in range(thb.level_count())
            for idx in active[level]
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    pts = [[x, y] for x in grid(21) for y in grid(21)]
    vals = [math.exp(-40 * ((p[0] - 0.15) ** 2 + (p[1] - 0.15) ** 2)) for p in pts]
    thb.fit(pts, vals)
    coarse = dm.TruncatedHierarchicalBSpline(dm.TensorBSpline([b, b], [0.0] * 16))
    coarse.fit(pts, vals)
    err_thb = sum((thb.evaluate(p) - v) ** 2 for p, v in zip(pts, vals))
    err_coarse = sum((coarse.evaluate(p) - v) ** 2 for p, v in zip(pts, vals))
    assert err_thb < err_coarse
    assert thb.basis_size() > coarse.basis_size()

    with pytest.raises(ValueError):
        thb.evaluate([1.5, 0.5])
    with pytest.raises(ValueError):
        thb.set_coefficient(0, [0, 0], 1.0)  # inactive once refined past
