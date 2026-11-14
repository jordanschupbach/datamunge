// CMD: just TARGET=rplot_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
using Datamunge;

// plot(x, y, type = "p") then abline() layered on afterward.
var scatter = RPlot.Plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed");
scatter.Abline(0.0, 2.0, new RGB(220, 38, 38), 1.5);
scatter.Title("plot() + abline()").XLabel("x").YLabel("y");
scatter.SaveSvg("csharp_rplot_scatter.svg");

// hist(): equal-width binning over the data range.
var histogram = RPlot.Hist([1.0, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts");
histogram.SaveSvg("csharp_rplot_hist.svg");

// barplot(): categorical positions with x tick labels.
RPlot.Barplot([23.0, 41.0, 12.0], ["A", "B", "C"]).Title("barplot()").SaveSvg("csharp_rplot_barplot.svg");

// boxplot(): Tukey five-number summary per group.
RPlot.Boxplot([[2.0, 4, 4, 4, 5, 5, 7, 9], [1.0, 2, 2, 2, 3, 3, 3, 3, 4, 20]], ["low variance", "has outlier"])
    .Title("boxplot()").SaveSvg("csharp_rplot_boxplot.svg");

// pie(): wedge areas proportional to value, axes hidden automatically.
RPlot.Pie([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"]).Title("pie()").SaveSvg("csharp_rplot_pie.svg");

// curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

// qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
double[] residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
RPlot.Qqnorm(residuals).Qqline(residuals).SaveSvg("csharp_rplot_qqnorm.svg");

// par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
RLayout.Create(1, 2).Add(scatter).Add(histogram).SaveSvg("csharp_rplot_layout.svg");

Console.WriteLine("Wrote 7 SVGs to csharp_rplot_*.svg");
