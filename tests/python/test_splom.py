import pytest


def _iris():
    from pydatamunge import datamunge

    return datamunge.DataFrame.iris()


def test_splom_selects_numeric_columns_and_renders_grid():
    from pydatamunge import datamunge

    matrix = datamunge.Splom(_iris())
    assert list(matrix.variables()) == ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]
    assert list(matrix.groups()) == []
    svg = matrix.title("Iris pairs").to_svg()
    assert svg.startswith("<?xml")
    assert svg.count("<svg") == 1 + 5 * 5
    assert "Iris pairs" in svg


def test_splom_setters_chain_and_group_by_groups():
    from pydatamunge import datamunge

    matrix = datamunge.Splom(_iris(), ["Sepal.Length", "Petal.Length"])
    same = matrix.group_by("Species").diagonal_histogram().bins(8).marker_size(3.0)
    assert same is matrix
    assert list(matrix.groups()) == ["setosa", "versicolor", "virginica"]
    panel = matrix.panel(0, 1)
    assert panel.has_custom_margins()
    assert panel.series().size() == 1
    assert panel.series()[0].point_colors.size() == 150
    assert matrix.to_svg().count("<svg") == 1 + 2 * 2


def test_splom_accepts_string_columns_and_rejects_missing_ones():
    from pydatamunge import datamunge

    matrix = datamunge.Splom(_iris(), ["Species", "Sepal.Length"])
    assert matrix.point_colors([datamunge.RGB()]) is matrix
    assert matrix.to_svg().count("<svg") == 1 + 2 * 2
    with pytest.raises(Exception):
        datamunge.Splom(_iris(), ["Sepal.Length", "nope"])
    with pytest.raises(Exception):
        datamunge.Splom(_iris(), ["Sepal.Length"])


def test_rlayout_title_and_plot_margins():
    from pydatamunge import datamunge

    p = datamunge.RPlot.create()
    p.points(datamunge.DVector([0.0, 1.0]), datamunge.DVector([0.0, 1.0]))
    p.margins(10.0, 5.0, 5.0, 10.0).tick_font_size(9.0)
    assert 'font-size="9"' in p.to_svg()
    layout = datamunge.RLayout.create(1, 1)
    layout.add(p).title("Grid")
    assert ">Grid</text>" in layout.to_svg()
