import math

from pydatamunge import datamunge as dm


def test_vector_helpers_and_bsr():
    dm.set_seed(3)
    u = dm.unif(2.0, 3.0, 100)
    assert len(u) == 100 and all(2.0 <= v < 3.0 for v in u)
    assert len(dm.rnorm(0.0, 1.0, 10)) == 10
    x = dm.seql(0.0, 1.0, 50)
    assert x[0] == 0.0 and x[-1] == 1.0

    y = [v**3 - v for v in x]
    fit = dm.BSR(x, y, 8)
    assert fit.basis_size() == 8
    assert math.isclose(fit.predict([0.5])[0], 0.125 - 0.5, abs_tol=1e-8)

    by_knots = dm.BSplineRegression(x, y, dm.seql(0.0, 1.0, 5))
    assert by_knots.basis_size() == 7


def test_rgba_and_chaining_on_a_temporary():
    plot = (
        dm.RPlot.create()
        .points([0.0, 1.0], [0.0, 1.0], "p", dm.RGBA(1, 2, 3, 0.25), 4.0)
        .title("T")
        .xlab("X")
        .ylab("Y")
    )
    assert plot.title_text() == "T" and plot.x_label_text() == "X"
    assert 'fill-opacity="0.25"' in plot.to_svg()
    assert dm.RGB().alpha == 1.0


def test_psr_fixed_lambda_and_gcv():
    x = dm.seql(0.0, 1.0, 60)
    line = [1.0 - 2.0 * v for v in x]
    fit = dm.PSR(x, line, 30, 1e4)  # the 2nd-order penalty leaves lines alone
    assert fit.basis_size() == 30 and fit._lambda() == 1e4  # SWIG renames the keyword
    assert math.isclose(fit.predict([0.25])[0], 0.5, abs_tol=1e-8)

    dm.set_seed(5)
    xs = dm.unif(0.0, 1.0, 200)
    noise = dm.rnorm(0.0, 0.1, 200)
    ys = [math.sin(6.0 * a) + e for a, e in zip(xs, noise)]
    auto = dm.PSplineRegression(xs, ys, 40)  # lambda chosen by GCV
    assert auto.gcv() <= dm.PSR(xs, ys, 40, 1e3).gcv()
    assert 2.0 < auto.effective_df() < 40.0
    assert dm.PSR(xs, ys, dm.seql(0.0, 1.0, 11)).basis_size() == 13
