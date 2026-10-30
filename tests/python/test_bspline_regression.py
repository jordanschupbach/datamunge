import math

import pytest

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


def test_point_palette_recycling_and_subtitle_preserve_ownership():
    import gc

    plot = dm.RPlot.create().points_colored(
        [1, 2, 3], [2, 3, 4], [dm.RGBA(10, 20, 30, 0.25), dm.RGB(40, 50, 60)]
    ).subtitle("Palette <test>")
    gc.collect()
    assert plot.thisown
    assert plot.subtitle_text() == "Palette <test>"
    svg = plot.to_svg()
    assert svg.count('fill="rgb(10,20,30)" fill-opacity="0.25"') == 2
    assert svg.count('fill="rgb(40,50,60)"') == 1
    assert "Palette &lt;test&gt;" in svg


def test_fluent_plot_labels_retain_the_temporary_owner():
    import gc

    plot = (dm.RPlot.create().set_title("Iris <test>").set_subtitle("Species")
            .set_xlab("Width").set_ylab("Length"))
    gc.collect()
    assert plot.thisown
    assert plot.title_text() == "Iris <test>"
    assert plot.subtitle_text() == "Species"
    assert plot.x_label_text() == "Width"
    assert plot.y_label_text() == "Length"
    assert plot.set_title("") is plot
    assert plot.title_text() == ""


def test_scatter_constructor_and_histogram_validation():
    plot = dm.RPlot([0.0, 1.0], [1.0, 2.0]).set_title("Samples").set_col("blue")
    assert plot.thisown
    assert plot.title_text() == "Samples"
    assert "<circle" in plot.to_svg()
    assert 'fill="rgb(0,0,255)"' in plot.to_svg()
    assert dm.RPlot().title_text() == "datamunge plot"
    with pytest.raises(ValueError):
        dm.RPlot([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        dm.RPlot.hist([float("nan")])
    with pytest.raises(ValueError):
        dm.RPlot.hist([float("inf")])
    with pytest.raises(ValueError):
        plot.set_col("unknown")
    dm.set_seed(42)
    uniform = dm.runif(0, 1, 10)
    dm.set_seed(42)
    assert uniform == dm.unif(0, 1, 10)
    with pytest.raises(ValueError):
        dm.runif(1, 0, 10)
