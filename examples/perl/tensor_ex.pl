# CMD: just TARGET=tensor_ex run-perl
# Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

sub shape_string { my ($tensor) = @_; return "[" . join(", ", @{ $tensor->shape() }) . "]" }

print "=================== Construction ===================\n";
print "zeros([2,3]): ", Datamunge::Tensor->zeros([2, 3]), "\n";
print "eye(3): ", Datamunge::Tensor->eye(3), "\n";

my $r = Datamunge::Tensor->arange(0.0, 12.0, 1.0)->reshape([3, 4]);
print "arange(0,12).reshape([3,4]): $r\n";

print "\n=================== Shape ops ===================\n";
print "transpose -> shape ", shape_string($r->transpose()), "\n";
print "slice(axis=1, start=1, stop=3): ", $r->slice(1, 1, 3), "\n";

print "\n=================== Broadcasting arithmetic ===================\n";
{
    my $col = Datamunge::Tensor->from_values([3, 1], [1.0, 2.0, 3.0]);
    my $row = Datamunge::Tensor->from_values([1, 4], [10.0, 20.0, 30.0, 40.0]);
    print "(3,1) + (1,4) -> ", $col->add($row), "\n";
}

print "\n=================== Reductions ===================\n";
print "r.sum() = ", fmt($r->sum()), ", r.mean() = ", fmt($r->mean()), "\n";
print "column means (axis=0): ", $r->mean_axis(0), "\n";

print "\n=================== Linear algebra ===================\n";
{
    my $a = Datamunge::Tensor->from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    my $b = Datamunge::Tensor->from_values([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    print "matmul(2x3, 3x2) -> ", $a->matmul($b), "\n";
}
{
    my $v1 = Datamunge::Tensor->from_values([3], [1.0, 2.0, 3.0]);
    my $v2 = Datamunge::Tensor->from_values([3], [4.0, 5.0, 6.0]);
    print "dot([1,2,3], [4,5,6]) = ", fmt($v1->dot($v2)), "\n";
    print "outer(v1, v2) -> ", $v1->outer($v2), "\n";
}

print "\n=================== Comparisons & masks ===================\n";
{
    my $mask = $r->greater_equal(Datamunge::Tensor->full([3, 4], 6.0));
    print "r >= 6 -> $mask\n";
    print "count(r >= 6) = ", fmt($mask->sum()), "\n";
}

print "\n=================== A real dataset as a Tensor ===================\n";
{
    my $iris = Datamunge::Data::iris();
    my $n = $iris->nrows();
    my @columns = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
    my @flat;
    for my $i (0 .. $n - 1) {
        push @flat, $iris->numeric_at($_, $i) for @columns;
    }
    my $x = Datamunge::Tensor->from_values([$n, 4], \@flat);
    print "iris feature tensor shape: ", shape_string($x), "\n";

    my $feature_means = $x->mean_axis(0);
    my $centered = $x->subtract($feature_means->reshape([1, 4]));
    my $scatter = $centered->transpose()->matmul($centered);
    print "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): $feature_means\n";
    print "(X-mean)^T (X-mean) [4x4 scatter matrix]: ", $scatter->to_string(16), "\n";

    my @species = map { $iris->string_at("Species", $_) } 0 .. $n - 1;
    my $species_tensor = Datamunge::Tensor->from_string_values([$n], \@species);
    my $setosa_mask = $species_tensor->equal(Datamunge::Tensor->from_string_values([1], ["setosa"]));
    print "setosa count = ", fmt($setosa_mask->sum()), " (of $n rows)\n";
}

# Formats a number like C#'s double.ToString(): shortest round-trip digits, E notation outside [1e-4, 1e15).
sub fmt {
    my ($x) = @_;
    return 'NaN' if $x != $x;
    return $x > 0 ? 'Infinity' : '-Infinity' if $x == 9**9**9 || $x == -9**9**9;
    return '0' if $x == 0;
    my $s;
    for my $p (0 .. 16) { $s = sprintf('%.*e', $p, $x); last if $s == $x }
    my ($m, $e) = $s =~ /^(-?[\d.]+)e([-+]\d+)$/;
    $m =~ s/\.?0+$// if $m =~ /\./;
    $e += 0;
    return sprintf('%sE%s%02d', $m, $e < 0 ? '-' : '+', abs $e) if $e < -4 || $e >= 15;
    my $neg = $m =~ s/^-//;
    my $digits = $m =~ s/\.//r;
    my $out;
    if ($e < 0) { $out = '0.' . ('0' x (-$e - 1)) . $digits }
    elsif (length($digits) > $e + 1) { $out = substr($digits, 0, $e + 1) . '.' . substr($digits, $e + 1) }
    else { $out = $digits . ('0' x ($e + 1 - length $digits)) }
    return ($neg ? '-' : '') . $out;
}
