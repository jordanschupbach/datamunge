"""Demonstrates dm.RPlot / dm.RLayout -- the R-base-graphics-style plotting library."""
import math
import os

from pydatamunge import datamunge as dm

os.makedirs("build/debug/examples", exist_ok=True)


class SineWave(dm.Callback):
    def call(self, x):
        return math.sin(x)


def rgb(r, g, b):
    color = dm.RGB()
    color.r, color.g, color.b = r, g, b
    return color


# plot(x, y, type = "p") then abline() layered on afterward.
scatter = dm.RPlot.plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed")
scatter.abline(0.0, 2.0, rgb(220, 38, 38), 1.5)
scatter.title("plot() + abline()").x_label("x").y_label("y")
scatter.save_svg("build/debug/examples/py_rplot_scatter.svg")

# hist(): equal-width binning over the data range.
samples = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7]
hist = dm.RPlot.hist(dm.DVector(samples), 6, "counts")
hist.save_svg("build/debug/examples/py_rplot_hist.svg")

# barplot(): categorical positions with x_tick_labels.
bars = dm.RPlot.barplot(dm.DVector([23.0, 41.0, 12.0]), dm.SVector(["A", "B", "C"]))
bars.title("barplot()")
bars.save_svg("build/debug/examples/py_rplot_barplot.svg")

# boxplot(): Tukey five-number summary per group.
groups = dm.DVectorVector([
    dm.DVector([2, 4, 4, 4, 5, 5, 7, 9]),
    dm.DVector([1, 2, 2, 2, 3, 3, 3, 3, 4, 20]),
])
box = dm.RPlot.boxplot(groups, dm.SVector(["low variance", "has outlier"]))
box.title("boxplot()")
box.save_svg("build/debug/examples/py_rplot_boxplot.svg")

# pie(): wedge areas proportional to value, axes hidden automatically.
pie = dm.RPlot.pie(dm.DVector([35.0, 25.0, 20.0, 20.0]), dm.SVector(["Q1", "Q2", "Q3", "Q4"]))
pie.title("pie()")
pie.save_svg("build/debug/examples/py_rplot_pie.svg")

# curve(): samples a Callback (a Python-subclassed director) over a range.
sine = SineWave()
curve = dm.RPlot.curve(sine, 0.0, 2.0 * math.pi, 200, "sin(x)")
curve.title("curve()")
curve.save_svg("build/debug/examples/py_rplot_curve.svg")

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
qq = dm.RPlot.qqnorm(dm.DVector(residuals))
qq.qqline(dm.DVector(residuals))
qq.save_svg("build/debug/examples/py_rplot_qqnorm.svg")

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
layout = dm.RLayout.create(1, 2)
layout.add(scatter)
layout.add(hist)
layout.save_svg("build/debug/examples/py_rplot_layout.svg")

print("Wrote 7 SVGs to build/debug/examples/py_rplot_*.svg")
