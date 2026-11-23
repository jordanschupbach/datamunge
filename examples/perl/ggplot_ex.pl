# CMD: just TARGET=ggplot_ex run-perl
# Demonstrates GgPlot -- the ggplot2-style grammar-of-graphics library.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
Datamunge::GgPlot->new($iris, "Sepal.Length", "Sepal.Width", "Species")
    ->geom_point()
    ->labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
    ->theme_minimal()
    ->save_svg("perl_ggplot_point.svg");

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
Datamunge::GgPlot->new($iris, "Sepal.Length", "Petal.Length")
    ->geom_point(Datamunge::Rgb->new(156, 163, 175), 2.5)
    ->geom_smooth()
    ->labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
    ->save_svg("perl_ggplot_smooth.svg");

# geom_bar(): counts a discrete column (stat = "count").
Datamunge::GgPlot->new($iris, "Species")
    ->geom_bar()
    ->labs("Observations per Species", "Species", "Count")
    ->theme_bw()
    ->save_svg("perl_ggplot_bar.svg");

# geom_boxplot(): grouped by a discrete x column.
Datamunge::GgPlot->new($iris, "Species", "Petal.Width")
    ->geom_boxplot()
    ->labs("Petal Width by Species", "Species", "Petal Width")
    ->save_svg("perl_ggplot_boxplot.svg");

# geom_histogram() + geom_density(): distribution of a single numeric column.
Datamunge::GgPlot->new($iris, "Sepal.Length")
    ->geom_histogram(20)
    ->labs("Distribution of Sepal Length", "Sepal Length", "Count")
    ->save_svg("perl_ggplot_histogram.svg");

Datamunge::GgPlot->new($iris, "Sepal.Length")
    ->geom_density()
    ->labs("Density of Sepal Length", "Sepal Length", "Density")
    ->save_svg("perl_ggplot_density.svg");

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
Datamunge::GgPlot->new($iris, "Petal.Length", "Petal.Width")
    ->geom_point()
    ->facet_wrap("Species")
    ->labs("Petal Dimensions", "Petal Length", "Petal Width")
    ->save_svg("perl_ggplot_facet.svg");

# theme_classic(): another built-in theme.
Datamunge::GgPlot->new($iris, "Sepal.Length", "Sepal.Width", "Species")
    ->geom_point()
    ->theme_classic()
    ->save_svg("perl_ggplot_classic.svg");

print "Wrote 7 SVGs to perl_ggplot_*.svg\n";
