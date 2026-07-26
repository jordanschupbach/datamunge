use strict;
use warnings;

use Datamunge;

# Fits a two-dimensional B-spline surface with the bs(x, y) formula term.
# Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
my (@x, @y, @z);
for my $iy (0 .. 7) {
  for my $ix (0 .. 7) {
    my $xv = $ix / 7.0;
    my $yv = $iy / 7.0;
    push @x, $xv;
    push @y, $yv;
    push @z, sin($xv) + $yv * $yv;
  }
}

my $data = Datamunge::DataFrame->new();
$data->add_numeric_column("x", \@x);
$data->add_numeric_column("y", \@y);
$data->add_numeric_column("z", \@z);

my $surface = Datamunge::LM->new($data, "z ~ bs(x, y)");
print "Fitted z ~ bs(x, y)\n";
$surface->print_summary();

my $new_points = Datamunge::DataFrame->new();
$new_points->add_numeric_column("x", [0.25, 0.75]);
$new_points->add_numeric_column("y", [0.50, 0.25]);
my $pred = $surface->predict($new_points);
printf("Predictions: [%.6f, %.6f]\n", $pred->[0], $pred->[1]);
