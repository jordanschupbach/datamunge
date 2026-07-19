use strict;
use warnings;

use Datamunge;

my @hp = (110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0);
my @wt = (2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44);
my @transmission = ("manual", "manual", "manual", "automatic", "automatic",
                     "automatic", "automatic", "automatic", "automatic", "automatic");
my @mpg = (21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2);

my $cars = Datamunge::DataFrame->new();
$cars->add_numeric_column("hp", \@hp);
$cars->add_numeric_column("wt", \@wt);
$cars->add_string_column("transmission", \@transmission);
$cars->add_numeric_column("mpg", \@mpg);

print "Fitting: mpg ~ hp + wt + transmission\n\n";
my $model = Datamunge::LM->new($cars, "mpg ~ hp + wt + transmission");
$model->print_summary();

print "\nSequential ANOVA:\n";
print $model->anova()->to_string(), "\n";

my $newcars = Datamunge::DataFrame->new();
$newcars->add_numeric_column("hp", [150.0, 90.0]);
$newcars->add_numeric_column("wt", [3.0, 2.5]);
$newcars->add_string_column("transmission", ["manual", "automatic"]);

my $frame = $model->predict_frame($newcars, "confidence");
print "\nPredictions with 95% confidence intervals:\n";
print $frame->to_string(), "\n";

$model->save_diagnostic_plots("lm_ex_diagnostics");
print "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n";
