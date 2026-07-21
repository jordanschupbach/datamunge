use strict;
use warnings;

use Datamunge;

my $sales = Datamunge::DataFrame->new();
$sales->add_string_column("region", ["west", "west", "east", "south", "south", "south"]);
$sales->add_string_column("product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]);
$sales->add_numeric_column("sales", [10, 10, 14, 8, 0, 11], [1, 1, 1, 1, 0, 1]);
$sales->add_string_column("quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0]);

print "raw data\n";
print $sales->to_string(), "\n\n";

my $cleaned = $sales->drop_duplicates(["region", "product", "sales", "quarter"]);
$cleaned->fill_null_string("quarter", "unknown");
$cleaned->fill_null_numeric("sales", 0.0);
print "after drop_duplicates + fill_null\n";
print $cleaned->to_string(), "\n\n";

my $selected = $cleaned->select(["region", "sales", "quarter"])->sort_by("sales", 0);
print "selected + sorted\n";
print $selected->to_string(), "\n\n";

my $grouped = $cleaned->group_by_sum(["region"], ["sales"])->sort_by("sales", 0);
print "group_by_sum(region)\n";
print $grouped->to_string(), "\n\n";

my $targets = Datamunge::DataFrame->new();
$targets->add_string_column("region", ["west", "east", "south"]);
$targets->add_numeric_column("target", [18, 12, 25]);
my $joined = $grouped->join($targets, "region", "region", "left");
print "joined with targets\n";
print $joined->to_string(), "\n\n";

my $shape = $cleaned->shape();
print "shape = (" . $shape->[0] . ", " . $shape->[1] . ")\n";
print "sales count = " . $cleaned->numeric_count("sales") . "\n";
print "sales nulls = " . $cleaned->numeric_null_count("sales") . "\n";
print "sales sum = " . $cleaned->numeric_sum("sales") . "\n";
print "sales mean = " . $cleaned->numeric_mean("sales") . "\n";
