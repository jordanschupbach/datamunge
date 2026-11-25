// CMD: just TARGET=ggplot_ex run-javascript
// Demonstrates datamunge.GGPlot -- the ggplot2-style grammar-of-graphics library.

const datamunge = require("../../index.js");

function rgb(r, g, b) {
  const color = new datamunge.RGB();
  color.r = r;
  color.g = g;
  color.b = b;
  return color;
}

const iris = datamunge.data.iris();

// ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
const scatter = new datamunge.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species");
scatter.geom_point();
scatter.labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width");
scatter.theme_minimal();
scatter.save_svg("javascript_ggplot_point.svg");

// + geom_smooth(): an lm() fit line, reusing stats::LM internally.
const smooth = new datamunge.GGPlot(iris, "Sepal.Length", "Petal.Length");
smooth.geom_point(rgb(156, 163, 175), 2.5);
smooth.geom_smooth();
smooth.labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length");
smooth.save_svg("javascript_ggplot_smooth.svg");

// geom_bar(): counts a discrete column (stat = "count").
const bar = new datamunge.GGPlot(iris, "Species");
bar.geom_bar();
bar.labs("Observations per Species", "Species", "Count");
bar.theme_bw();
bar.save_svg("javascript_ggplot_bar.svg");

// geom_boxplot(): grouped by a discrete x column.
const box = new datamunge.GGPlot(iris, "Species", "Petal.Width");
box.geom_boxplot();
box.labs("Petal Width by Species", "Species", "Petal Width");
box.save_svg("javascript_ggplot_boxplot.svg");

// geom_histogram() + geom_density(): distribution of a single numeric column.
const hist = new datamunge.GGPlot(iris, "Sepal.Length");
hist.geom_histogram(20);
hist.labs("Distribution of Sepal Length", "Sepal Length", "Count");
hist.save_svg("javascript_ggplot_histogram.svg");

const density = new datamunge.GGPlot(iris, "Sepal.Length");
density.geom_density();
density.labs("Density of Sepal Length", "Sepal Length", "Density");
density.save_svg("javascript_ggplot_density.svg");

// facet_wrap(): one panel per Species, composed via RLayout under the hood.
const faceted = new datamunge.GGPlot(iris, "Petal.Length", "Petal.Width");
faceted.geom_point();
faceted.facet_wrap("Species");
faceted.labs("Petal Dimensions", "Petal Length", "Petal Width");
faceted.save_svg("javascript_ggplot_facet.svg");

// scale_color_manual(): override the default discrete palette (a plain array of RGB values).
const customColors = new datamunge.GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species");
customColors.geom_point();
customColors.scale_color_manual([rgb(16, 185, 129), rgb(245, 158, 11), rgb(99, 102, 241)]);
customColors.theme_classic();
customColors.save_svg("javascript_ggplot_custom_colors.svg");

console.log("Wrote 7 SVGs to javascript_ggplot_*.svg");
