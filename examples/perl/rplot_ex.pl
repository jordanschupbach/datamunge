use strict;
use warnings;

use Datamunge;

# Demonstrates Datamunge::RPlot / Datamunge::RLayout -- the R-base-graphics-style plotting library.

# curve() samples a Callback (a SWIG director). The Perl binding DOES support directors, so
# unlike the Lua template this example subclasses Callback directly.
package SineWave;
our @ISA = ("Datamunge::Callback");

sub call {
  my ($self, $x) = @_;
  return sin($x);
}

package main;

sub rgb {
  my ($r, $g, $b) = @_;
  my $c = Datamunge::RGB->new();
  $c->swig_r_set($r);
  $c->swig_g_set($g);
  $c->swig_b_set($b);
  return $c;
}

my $pi = 3.14159265358979;

# plot(x, y, type = "p") then abline() layered on afterward.
my $scatter = Datamunge::RPlot::plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], "p", "observed");
$scatter->abline(0.0, 2.0, rgb(220, 38, 38), 1.5);
$scatter->title("plot() + abline()");
$scatter->x_label("x");
$scatter->y_label("y");
$scatter->save_svg("perl_rplot_scatter.svg");

# hist(): equal-width binning over the data range.
my $hist = Datamunge::RPlot::hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7], 6, "counts");
$hist->save_svg("perl_rplot_hist.svg");

# barplot(): categorical positions with x tick labels.
my $bars = Datamunge::RPlot::barplot([23.0, 41.0, 12.0], ["A", "B", "C"]);
$bars->title("barplot()");
$bars->save_svg("perl_rplot_barplot.svg");

# boxplot(): Tukey five-number summary per group.
my $groups = Datamunge::DVectorVector->new();
$groups->push([2, 4, 4, 4, 5, 5, 7, 9]);
$groups->push([1, 2, 2, 2, 3, 3, 3, 3, 4, 20]);
my $box = Datamunge::RPlot::boxplot($groups, ["low variance", "has outlier"]);
$box->title("boxplot()");
$box->save_svg("perl_rplot_boxplot.svg");

# pie(): wedge areas proportional to value, axes hidden automatically.
my $pie = Datamunge::RPlot::pie([35.0, 25.0, 20.0, 20.0], ["Q1", "Q2", "Q3", "Q4"]);
$pie->title("pie()");
$pie->save_svg("perl_rplot_pie.svg");

# curve(): samples a Callback (a Perl-subclassed director) over a range.
my $sine = SineWave->new();
my $curve = Datamunge::RPlot::curve($sine, 0.0, 2.0 * $pi, 200, "sin(x)");
$curve->title("curve()");
$curve->save_svg("perl_rplot_curve.svg");

# qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
my @residuals = (-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3);
my $qq = Datamunge::RPlot::qqnorm(\@residuals);
$qq->qqline(\@residuals);
$qq->save_svg("perl_rplot_qqnorm.svg");

# par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
my $layout = Datamunge::RLayout::create(1, 2);
$layout->add($scatter);
$layout->add($hist);
$layout->save_svg("perl_rplot_layout.svg");

print "Wrote 7 SVGs to perl_rplot_*.svg\n";
