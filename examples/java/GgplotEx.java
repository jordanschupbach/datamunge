// CMD: just TARGET=ggplot_ex run-java
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
import datamunge.GGPlot;
import datamunge.RGB;
import datamunge.data.Data;

public class GgplotEx {
  public static void main(String[] args) {
    var iris = Data.iris();

    // Each GGPlot method returns the plot, so a ggplot2 "+ layer" chain becomes a method chain.
    // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
    new GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
        .geomPoint()
        .labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
        .themeMinimal()
        .saveSvg("java_ggplot_point.svg");

    // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
    new GGPlot(iris, "Sepal.Length", "Petal.Length")
        .geomPoint(new RGB(156, 163, 175), 2.5)
        .geomSmooth()
        .labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
        .saveSvg("java_ggplot_smooth.svg");

    // geom_bar(): counts a discrete column (stat = "count").
    new GGPlot(iris, "Species")
        .geomBar()
        .labs("Observations per Species", "Species", "Count")
        .themeBw()
        .saveSvg("java_ggplot_bar.svg");

    // geom_boxplot(): grouped by a discrete x column.
    new GGPlot(iris, "Species", "Petal.Width")
        .geomBoxplot()
        .labs("Petal Width by Species", "Species", "Petal Width")
        .saveSvg("java_ggplot_boxplot.svg");

    // geom_histogram() + geom_density(): distribution of a single numeric column.
    new GGPlot(iris, "Sepal.Length")
        .geomHistogram(20)
        .labs("Distribution of Sepal Length", "Sepal Length", "Count")
        .saveSvg("java_ggplot_histogram.svg");

    new GGPlot(iris, "Sepal.Length")
        .geomDensity()
        .labs("Density of Sepal Length", "Sepal Length", "Density")
        .saveSvg("java_ggplot_density.svg");

    // facet_wrap(): one panel per Species, composed via RLayout under the hood.
    new GGPlot(iris, "Petal.Length", "Petal.Width")
        .geomPoint()
        .facetWrap("Species")
        .labs("Petal Dimensions", "Petal Length", "Petal Width")
        .saveSvg("java_ggplot_facet.svg");

    // theme_classic(): another built-in theme.
    new GGPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
        .geomPoint()
        .themeClassic()
        .saveSvg("java_ggplot_classic.svg");

    System.out.println("Wrote 7 SVGs to java_ggplot_*.svg");
  }
}
