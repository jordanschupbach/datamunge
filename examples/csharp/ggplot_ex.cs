// CMD: just TARGET=ggplot_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
using Datamunge;

var iris = Data.Iris()!;

// ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
new GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
    .GeomPoint()
    .Labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
    .ThemeMinimal()
    .SaveSvg("csharp_ggplot_point.svg");

// + geom_smooth(): an lm() fit line, reusing stats::LM internally.
new GGPlot(iris, "Sepal.Length", "Petal.Length")
    .GeomPoint(new RGB(156, 163, 175), 2.5)
    .GeomSmooth()
    .Labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
    .SaveSvg("csharp_ggplot_smooth.svg");

// geom_bar(): counts a discrete column (stat = "count").
new GGPlot(iris, "Species")
    .GeomBar()
    .Labs("Observations per Species", "Species", "Count")
    .ThemeBw()
    .SaveSvg("csharp_ggplot_bar.svg");

// geom_boxplot(): grouped by a discrete x column.
new GGPlot(iris, "Species", "Petal.Width")
    .GeomBoxplot()
    .Labs("Petal Width by Species", "Species", "Petal Width")
    .SaveSvg("csharp_ggplot_boxplot.svg");

// geom_histogram() + geom_density(): distribution of a single numeric column.
new GGPlot(iris, "Sepal.Length")
    .GeomHistogram(20)
    .Labs("Distribution of Sepal Length", "Sepal Length", "Count")
    .SaveSvg("csharp_ggplot_histogram.svg");

new GGPlot(iris, "Sepal.Length")
    .GeomDensity()
    .Labs("Density of Sepal Length", "Sepal Length", "Density")
    .SaveSvg("csharp_ggplot_density.svg");

// facet_wrap(): one panel per Species, composed via RLayout under the hood.
new GGPlot(iris, "Petal.Length", "Petal.Width")
    .GeomPoint()
    .FacetWrap("Species")
    .Labs("Petal Dimensions", "Petal Length", "Petal Width")
    .SaveSvg("csharp_ggplot_facet.svg");

// theme_classic(): another built-in theme.
new GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
    .GeomPoint()
    .ThemeClassic()
    .SaveSvg("csharp_ggplot_classic.svg");

Console.WriteLine("Wrote 7 SVGs to csharp_ggplot_*.svg");
