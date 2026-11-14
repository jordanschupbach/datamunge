# CMD: just TARGET=rplot_ex run-julia

# R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
using Datamunge

# plot(x, y, type = "p") then abline() layered on afterward.
const scatter = plot(RPlot, [1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed")
abline(scatter, 0.0, 2.0, RGB(220, 38, 38), 1.5)
y_label(x_label(title(scatter, "plot() + abline()"), "x"), "y")
save_svg(scatter, "julia_rplot_scatter.svg")

# hist(): equal-width binning over the data range.
const histogram = hist(RPlot, [1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts")
save_svg(histogram, "julia_rplot_hist.svg")

# barplot(): categorical positions with x tick labels.
let bars = barplot(RPlot, [23.0, 41.0, 12.0], ["A", "B", "C"])
    title(bars, "barplot()")
    save_svg(bars, "julia_rplot_barplot.svg")
end

# boxplot(): Tukey five-number summary per group.
let box = boxplot(RPlot, [[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]],
                  ["low variance", "has outlier"])
    title(box, "boxplot()")
    save_svg(box, "julia_rplot_boxplot.svg")
end

# pie(): wedge areas proportional to value, axes hidden automatically.
let pie_chart = pie(RPlot, [35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"])
    title(pie_chart, "pie()")
    save_svg(pie_chart, "julia_rplot_pie.svg")
end

# curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
let residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
    qq = qqnorm(RPlot, residuals)
    qqline(qq, residuals)
    save_svg(qq, "julia_rplot_qqnorm.svg")
end

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
let layout = create(RLayout, 1, 2)
    add(layout, scatter)
    add(layout, histogram)
    save_svg(layout, "julia_rplot_layout.svg")
end

println("Wrote 7 SVGs to julia_rplot_*.svg")
