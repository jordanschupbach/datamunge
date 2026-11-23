<?php
// CMD: just TARGET=ggplot_ex run-php
// Demonstrates GgPlot -- the ggplot2-style grammar-of-graphics library.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{GgPlot, Rgb};
use function Datamunge\Data\iris;

$iris = iris();

// ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
(new GgPlot($iris, 'Sepal.Length', 'Sepal.Width', 'Species'))
    ->geomPoint()
    ->labs('Iris Sepal Dimensions', 'Sepal Length', 'Sepal Width')
    ->themeMinimal()
    ->saveSvg('php_ggplot_point.svg');

// + geom_smooth(): an lm() fit line, reusing stats::LM internally.
(new GgPlot($iris, 'Sepal.Length', 'Petal.Length'))
    ->geomPoint(new Rgb(156, 163, 175), 2.5)
    ->geomSmooth()
    ->labs('Petal Length vs Sepal Length With a Linear Fit', 'Sepal Length', 'Petal Length')
    ->saveSvg('php_ggplot_smooth.svg');

// geom_bar(): counts a discrete column (stat = "count").
(new GgPlot($iris, 'Species'))
    ->geomBar()
    ->labs('Observations per Species', 'Species', 'Count')
    ->themeBw()
    ->saveSvg('php_ggplot_bar.svg');

// geom_boxplot(): grouped by a discrete x column.
(new GgPlot($iris, 'Species', 'Petal.Width'))
    ->geomBoxplot()
    ->labs('Petal Width by Species', 'Species', 'Petal Width')
    ->saveSvg('php_ggplot_boxplot.svg');

// geom_histogram() + geom_density(): distribution of a single numeric column.
(new GgPlot($iris, 'Sepal.Length'))
    ->geomHistogram(20)
    ->labs('Distribution of Sepal Length', 'Sepal Length', 'Count')
    ->saveSvg('php_ggplot_histogram.svg');

(new GgPlot($iris, 'Sepal.Length'))
    ->geomDensity()
    ->labs('Density of Sepal Length', 'Sepal Length', 'Density')
    ->saveSvg('php_ggplot_density.svg');

// facet_wrap(): one panel per Species, composed via RLayout under the hood.
(new GgPlot($iris, 'Petal.Length', 'Petal.Width'))
    ->geomPoint()
    ->facetWrap('Species')
    ->labs('Petal Dimensions', 'Petal Length', 'Petal Width')
    ->saveSvg('php_ggplot_facet.svg');

// theme_classic(): another built-in theme.
(new GgPlot($iris, 'Sepal.Length', 'Sepal.Width', 'Species'))
    ->geomPoint()
    ->themeClassic()
    ->saveSvg('php_ggplot_classic.svg');

echo "Wrote 7 SVGs to php_ggplot_*.svg\n";
