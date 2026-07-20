"""Demonstrates dm.GGPlot -- the ggplot2-style grammar-of-graphics library."""
import os

from pydatamunge import datamunge as dm

os.makedirs("build/debug/examples", exist_ok=True)


def rgb(r, g, b):
    color = dm.RGB()
    color.r, color.g, color.b = r, g, b
    return color


iris = dm.DataFrame.iris()

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
scatter = dm.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
scatter.geom_point()
scatter.labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
scatter.theme_minimal()
scatter.save_svg("build/debug/examples/py_ggplot_point.svg")

# + geom_smooth(): an lm() fit line per group, reusing stats::LM internally.
smooth = dm.GGPlot(iris, "Sepal.Length", "Petal.Length")
smooth.geom_point(rgb(156, 163, 175), 2.5)
smooth.geom_smooth()
smooth.labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
smooth.save_svg("build/debug/examples/py_ggplot_smooth.svg")

# geom_bar(): counts a discrete column (stat = "count").
bar = dm.GGPlot(iris, "Species")
bar.geom_bar()
bar.labs("Observations per Species", "Species", "Count")
bar.theme_bw()
bar.save_svg("build/debug/examples/py_ggplot_bar.svg")

# geom_boxplot(): grouped by a discrete x column.
box = dm.GGPlot(iris, "Species", "Petal.Width")
box.geom_boxplot()
box.labs("Petal Width by Species", "Species", "Petal Width")
box.save_svg("build/debug/examples/py_ggplot_boxplot.svg")

# geom_histogram() + geom_density(): distribution of a single numeric column.
hist = dm.GGPlot(iris, "Sepal.Length")
hist.geom_histogram(20)
hist.labs("Distribution of Sepal Length", "Sepal Length", "Count")
hist.save_svg("build/debug/examples/py_ggplot_histogram.svg")

density = dm.GGPlot(iris, "Sepal.Length")
density.geom_density()
density.labs("Density of Sepal Length", "Sepal Length", "Density")
density.save_svg("build/debug/examples/py_ggplot_density.svg")

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
faceted = dm.GGPlot(iris, "Petal.Length", "Petal.Width")
faceted.geom_point()
faceted.facet_wrap("Species")
faceted.labs("Petal Dimensions", "Petal Length", "Petal Width")
faceted.save_svg("build/debug/examples/py_ggplot_facet.svg")

# scale_color_manual(): override the default discrete palette.
custom_colors = dm.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
custom_colors.geom_point()
custom_colors.scale_color_manual(dm.RGBVector([rgb(16, 185, 129), rgb(245, 158, 11), rgb(99, 102, 241)]))
custom_colors.theme_classic()
custom_colors.save_svg("build/debug/examples/py_ggplot_custom_colors.svg")

print("Wrote 7 SVGs to build/debug/examples/py_ggplot_*.svg")
