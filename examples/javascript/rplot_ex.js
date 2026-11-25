// CMD: just TARGET=rplot_ex run-javascript
// Demonstrates datamunge.RPlot / datamunge.RLayout -- the R-base-graphics-style plotting library.

const datamunge = require("../../index.js");

// A JavaScript subclass of the datamunge.Callback director, sampled by RPlot.curve().
class SineWave extends datamunge.Callback {
  call(x) {
    return Math.sin(x);
  }
}

function rgb(r, g, b) {
  const color = new datamunge.RGB();
  color.r = r;
  color.g = g;
  color.b = b;
  return color;
}

// plot(x, y, type = "p") then abline() layered on afterward.
const scatter = datamunge.RPlot.plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed");
scatter.abline(0.0, 2.0, rgb(220, 38, 38), 1.5);
scatter.title("plot() + abline()").x_label("x").y_label("y");
scatter.save_svg("javascript_rplot_scatter.svg");

// hist(): equal-width binning over the data range.
const hist = datamunge.RPlot.hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts");
hist.save_svg("javascript_rplot_hist.svg");

// barplot(): categorical positions with x tick labels.
const bars = datamunge.RPlot.barplot([23.0, 41.0, 12.0], ["A", "B", "C"]);
bars.title("barplot()");
bars.save_svg("javascript_rplot_barplot.svg");

// boxplot(): Tukey five-number summary per group.
const groups = [
  [2, 4, 4, 4, 5, 5, 7, 9],
  [1, 2, 2, 2, 3, 3, 3, 3, 4, 20],
];
const box = datamunge.RPlot.boxplot(groups, ["low variance", "has outlier"]);
box.title("boxplot()");
box.save_svg("javascript_rplot_boxplot.svg");

// pie(): wedge areas proportional to value, axes hidden automatically.
const pie = datamunge.RPlot.pie([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"]);
pie.title("pie()");
pie.save_svg("javascript_rplot_pie.svg");

// curve(): samples a Callback (a JavaScript-subclassed director) over a range.
const sine = new SineWave();
const curve = datamunge.RPlot.curve(sine, 0.0, 2.0 * Math.PI, 200, "sin(x)");
curve.title("curve()");
curve.save_svg("javascript_rplot_curve.svg");

// qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
const residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
const qq = datamunge.RPlot.qqnorm(residuals);
qq.qqline(residuals);
qq.save_svg("javascript_rplot_qqnorm.svg");

// par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
const layout = datamunge.RLayout.create(1, 2);
layout.add(scatter);
layout.add(hist);
layout.save_svg("javascript_rplot_layout.svg");

console.log("Wrote 8 SVGs to javascript_rplot_*.svg");
