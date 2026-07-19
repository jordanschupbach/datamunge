use strict;
use warnings;

use Datamunge;

# Plain Perl arrayrefs are accepted directly for std::vector<size_t> (shape/index)
# parameters -- no SizeVector wrapper needed.

print "=================== Construction ===================\n";
my $z = Datamunge::Tensor::zeros([2, 3]);
print "zeros([2,3]): " . $z->to_string() . "\n";

my $eye = Datamunge::Tensor::eye(3);
print "eye(3): " . $eye->to_string() . "\n";

my $r = Datamunge::Tensor::arange(0.0, 12.0, 1.0)->reshape([3, 4]);
print "arange(0,12).reshape([3,4]): " . $r->to_string() . "\n";

print "\n=================== Shape ops ===================\n";
my $rt = $r->transpose();
print "transpose -> shape [" . join(", ", @{$rt->shape()}) . "]\n";
my $sliced = $r->slice(1, 1, 3);
print "slice(axis=1, start=1, stop=3): " . $sliced->to_string() . "\n";

print "\n=================== Broadcasting arithmetic ===================\n";
my $col = Datamunge::Tensor::from_values([3, 1], [1, 2, 3]);
my $row = Datamunge::Tensor::from_values([1, 4], [10, 20, 30, 40]);
my $broadcast_sum = $col->add($row);
print "(3,1) + (1,4) -> " . $broadcast_sum->to_string() . "\n";

print "\n=================== Reductions ===================\n";
print "r.sum() = " . $r->sum() . ", r.mean() = " . $r->mean() . "\n";
my $col_means = $r->mean_axis(0);
print "column means (axis=0): " . $col_means->to_string() . "\n";

print "\n=================== Linear algebra ===================\n";
my $a = Datamunge::Tensor::from_values([2, 3], [1, 2, 3, 4, 5, 6]);
my $b = Datamunge::Tensor::from_values([3, 2], [7, 8, 9, 10, 11, 12]);
print "matmul(2x3, 3x2) -> " . $a->matmul($b)->to_string() . "\n";

my $v1 = Datamunge::Tensor::from_values([3], [1, 2, 3]);
my $v2 = Datamunge::Tensor::from_values([3], [4, 5, 6]);
print "dot([1,2,3], [4,5,6]) = " . $v1->dot($v2) . "\n";
print "outer(v1, v2) -> " . $v1->outer($v2)->to_string() . "\n";

print "\n=================== Comparisons & masks ===================\n";
my $mask = $r->greater_equal(Datamunge::Tensor::full([3, 4], 6.0));
print "r >= 6 -> " . $mask->to_string() . "\n";
print "count(r >= 6) = " . $mask->sum() . "\n";

print "\n=================== A real dataset as a Tensor ===================\n";
my $iris = Datamunge::DataFrame::iris();
my @flat;
for (my $i = 0; $i < $iris->nrows(); $i++) {
  push @flat, $iris->numeric_at("Sepal.Length", $i);
  push @flat, $iris->numeric_at("Sepal.Width", $i);
  push @flat, $iris->numeric_at("Petal.Length", $i);
  push @flat, $iris->numeric_at("Petal.Width", $i);
}
my $x = Datamunge::Tensor::from_values([$iris->nrows(), 4], \@flat);
print "iris feature tensor shape: [" . join(", ", @{$x->shape()}) . "]\n";

my $feature_means = $x->mean_axis(0);
my $centered = $x->subtract($feature_means->reshape([1, 4]));
my $scatter = $centered->transpose()->matmul($centered);
print "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " . $feature_means->to_string() . "\n";
print "(X-mean)^T (X-mean) [4x4 scatter matrix]: " . $scatter->to_string(16) . "\n";

my @species = map { $iris->string_at("Species", $_) } (0 .. $iris->nrows() - 1);
my $species_tensor = Datamunge::Tensor::from_string_values([$iris->nrows()], \@species);
my $setosa_mask = $species_tensor->equal(Datamunge::Tensor::from_string_values([1], ["setosa"]));
print "setosa count = " . $setosa_mask->sum() . " (of " . $iris->nrows() . " rows)\n";
