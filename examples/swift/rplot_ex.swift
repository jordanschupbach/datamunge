// CMD: just TARGET=rplot_ex run-swift
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

do {
    // plot(x, y, type = "p") then abline() layered on afterward.
    let scatter = try RPlot.plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed")
    _ = try scatter.abline(0.0, 2.0, try Rgb(220, 38, 38), 1.5)
    _ = try scatter.title("plot() + abline()").xLabel("x").yLabel("y")
    try scatter.saveSvg("swift_rplot_scatter.svg")

    // hist(): equal-width binning over the data range.
    let histogram = try RPlot.hist([1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts")
    try histogram.saveSvg("swift_rplot_hist.svg")

    // barplot(): categorical positions with x tick labels.
    try RPlot.barplot([23.0, 41.0, 12.0], ["A", "B", "C"]).title("barplot()").saveSvg("swift_rplot_barplot.svg")

    // boxplot(): Tukey five-number summary per group.
    try RPlot.boxplot([[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]], ["low variance", "has outlier"])
        .title("boxplot()").saveSvg("swift_rplot_boxplot.svg")

    // pie(): wedge areas proportional to value, axes hidden automatically.
    try RPlot.pie([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"]).title("pie()").saveSvg("swift_rplot_pie.svg")

    // curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

    // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
    let residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
    try RPlot.qqnorm(residuals).qqline(residuals).saveSvg("swift_rplot_qqnorm.svg")

    // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
    try RLayout.create(1, 2).add(scatter).add(histogram).saveSvg("swift_rplot_layout.svg")

    print("Wrote 7 SVGs to swift_rplot_*.svg")
} catch {
    print(error)
    exit(1)
}
