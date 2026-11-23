// CMD: just TARGET=ggplot_ex run-dart
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
import 'package:datamunge/datamunge.dart';

void main() {
  final iris = Data.iris()!;

  // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
  GgPlot(iris, 'Sepal.Length', 'Sepal.Width', 'Species')
      .geomPoint()
      .labs('Iris Sepal Dimensions', 'Sepal Length', 'Sepal Width')
      .themeMinimal()
      .saveSvg('dart_ggplot_point.svg');

  // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
  GgPlot(iris, 'Sepal.Length', 'Petal.Length')
      .geomPoint(Rgb(156, 163, 175), 2.5)
      .geomSmooth()
      .labs('Petal Length vs Sepal Length With a Linear Fit', 'Sepal Length', 'Petal Length')
      .saveSvg('dart_ggplot_smooth.svg');

  // geom_bar(): counts a discrete column (stat = "count").
  GgPlot(iris, 'Species')
      .geomBar()
      .labs('Observations per Species', 'Species', 'Count')
      .themeBw()
      .saveSvg('dart_ggplot_bar.svg');

  // geom_boxplot(): grouped by a discrete x column.
  GgPlot(iris, 'Species', 'Petal.Width')
      .geomBoxplot()
      .labs('Petal Width by Species', 'Species', 'Petal Width')
      .saveSvg('dart_ggplot_boxplot.svg');

  // geom_histogram() + geom_density(): distribution of a single numeric column.
  GgPlot(iris, 'Sepal.Length')
      .geomHistogram(20)
      .labs('Distribution of Sepal Length', 'Sepal Length', 'Count')
      .saveSvg('dart_ggplot_histogram.svg');

  GgPlot(iris, 'Sepal.Length')
      .geomDensity()
      .labs('Density of Sepal Length', 'Sepal Length', 'Density')
      .saveSvg('dart_ggplot_density.svg');

  // facet_wrap(): one panel per Species, composed via RLayout under the hood.
  GgPlot(iris, 'Petal.Length', 'Petal.Width')
      .geomPoint()
      .facetWrap('Species')
      .labs('Petal Dimensions', 'Petal Length', 'Petal Width')
      .saveSvg('dart_ggplot_facet.svg');

  // theme_classic(): another built-in theme.
  GgPlot(iris, 'Sepal.Length', 'Sepal.Width', 'Species')
      .geomPoint()
      .themeClassic()
      .saveSvg('dart_ggplot_classic.svg');

  print('Wrote 7 SVGs to dart_ggplot_*.svg');
}
