"""Native color parsing and RGB compatibility through generated SWIG constructors."""
import pytest
from pydatamunge import datamunge as dm


def test_rgba_constructor_parses_names_and_hex_without_borrowing():
    color = dm.RGBA("#ff448855")
    assert (color.r, color.g, color.b) == (255, 68, 136)
    assert color.alpha == pytest.approx(85 / 255)
    assert dm.RGB("GrEeN").g == 128
    assert dm.RGB(10, 20, 30).alpha == 1
    assert dm.RGB(10, 20, 30, 0.25).alpha == 0.25
    plot = dm.RPlot().lines([0, 1], [1, 2], "fit", color)
    del color
    svg = plot.to_svg()
    assert "rgb(255,68,136)" in svg
    assert 'opacity="0.333333"' in svg
    with pytest.raises(ValueError, match="color"):
        dm.RGB("#112233zz")


def test_axis_alias_chains_keep_the_owning_plot_alive_and_translate_errors():
    import gc
    plot = dm.RPlot().xlim(0, 1).ylim(-4, 4)
    gc.collect()
    assert (plot.x_min(), plot.x_max(), plot.y_min(), plot.y_max()) == (0, 1, -4, 4)
    assert plot.xlim(0, 2) is plot
    assert plot.ylim(-5, 5) is plot
    with pytest.raises(ValueError):
        plot.xlim(2, 1)
    with pytest.raises(ValueError):
        plot.ylim(0, float("nan"))
    assert (plot.x_max(), plot.y_max()) == (2, 5)
