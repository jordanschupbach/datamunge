# CMD: just TARGET=rplot_ex run-ruby
# R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
require "datamunge"

RPlot = Datamunge::RPlot

# plot(x, y, type = "p") then abline() layered on afterward.
scatter = RPlot.plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed")
scatter.abline(0.0, 2.0, Datamunge::RGB.new(220, 38, 38), 1.5)
scatter.title("plot() + abline()").x_label("x").y_label("y")
scatter.save_svg("ruby_rplot_scatter.svg")

# hist(): equal-width binning over the data range.
histogram = RPlot.hist([1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts")
histogram.save_svg("ruby_rplot_hist.svg")

# barplot(): categorical positions with x tick labels.
RPlot.barplot([23.0, 41.0, 12.0], %w[A B C]).title("barplot()").save_svg("ruby_rplot_barplot.svg")

# boxplot(): Tukey five-number summary per group.
RPlot.boxplot([[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]], ["low variance", "has outlier"])
     .title("boxplot()").save_svg("ruby_rplot_boxplot.svg")

# pie(): wedge areas proportional to value, axes hidden automatically.
RPlot.pie([35.0, 25.0, 20.0, 20.0], %w[Q1 Q2 Q3 Q4]).title("pie()").save_svg("ruby_rplot_pie.svg")

# curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3]
RPlot.qqnorm(residuals).qqline(residuals).save_svg("ruby_rplot_qqnorm.svg")

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
Datamunge::RLayout.create(1, 2).add(scatter).add(histogram).save_svg("ruby_rplot_layout.svg")

puts "Wrote 7 SVGs to ruby_rplot_*.svg"
