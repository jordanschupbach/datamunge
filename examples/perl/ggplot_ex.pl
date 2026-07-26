use strict;
use warnings;

use Datamunge;

# Demonstrates Datamunge::GGPlot -- the ggplot2-style grammar-of-graphics library.

sub rgb {
  my ($r, $g, $b) = @_;
  my $c = Datamunge::RGB->new();
  $c->swig_r_set($r);
  $c->swig_g_set($g);
  $c->swig_b_set($b);
  return $c;
}

my $iris = Datamunge::DataFrame::iris();

# ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
my $scatter = Datamunge::GGPlot->new($iris, "Sepal.Length", "Sepal.Width", "Species");
$scatter->geom_point();
$scatter->labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width");
$scatter->theme_minimal();
$scatter->save_svg("perl_ggplot_point.svg");

# + geom_smooth(): an lm() fit line, reusing stats::LM internally.
my $smooth = Datamunge::GGPlot->new($iris, "Sepal.Length", "Petal.Length");
$smooth->geom_point(rgb(156, 163, 175), 2.5);
$smooth->geom_smooth();
$smooth->labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length");
$smooth->save_svg("perl_ggplot_smooth.svg");

# geom_bar(): counts a discrete column (stat = "count").
my $bar = Datamunge::GGPlot->new($iris, "Species");
$bar->geom_bar();
$bar->labs("Observations per Species", "Species", "Count");
$bar->theme_bw();
$bar->save_svg("perl_ggplot_bar.svg");

# geom_boxplot(): grouped by a discrete x column.
my $box = Datamunge::GGPlot->new($iris, "Species", "Petal.Width");
$box->geom_boxplot();
$box->labs("Petal Width by Species", "Species", "Petal Width");
$box->save_svg("perl_ggplot_boxplot.svg");

# geom_histogram() + geom_density(): distribution of a single numeric column.
my $hist = Datamunge::GGPlot->new($iris, "Sepal.Length");
$hist->geom_histogram(20);
$hist->labs("Distribution of Sepal Length", "Sepal Length", "Count");
$hist->save_svg("perl_ggplot_histogram.svg");

my $density = Datamunge::GGPlot->new($iris, "Sepal.Length");
$density->geom_density();
$density->labs("Density of Sepal Length", "Sepal Length", "Density");
$density->save_svg("perl_ggplot_density.svg");

# facet_wrap(): one panel per Species, composed via RLayout under the hood.
my $faceted = Datamunge::GGPlot->new($iris, "Petal.Length", "Petal.Width");
$faceted->geom_point();
$faceted->facet_wrap("Species");
$faceted->labs("Petal Dimensions", "Petal Length", "Petal Width");
$faceted->save_svg("perl_ggplot_facet.svg");

# theme_classic(): another built-in theme. (scale_color_manual is omitted here -- the
# std::vector<RGB> overload it needs has no template (RGBVector) in the Perl binding's
# exposed surface.)
my $classic = Datamunge::GGPlot->new($iris, "Sepal.Length", "Sepal.Width", "Species");
$classic->geom_point();
$classic->theme_classic();
$classic->save_svg("perl_ggplot_classic.svg");

print "Wrote 7 SVGs to perl_ggplot_*.svg\n";
