# CMD: just TARGET=stl_ex run-perl
# C++ std::vector and std::pair values in Perl: vectors are array references, pairs array references of two.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $n = 100;
my @v = map { $_ * 1.5 } 0 .. $n - 1;
print "$_\n" for @v;
my @v2 = map { int($_ * 1.5) } 0 .. $n - 1;
print "$_\n" for @v2;

my @p = (3, 4);
print "p: ($p[0], $p[1])\n";

my @p2 = (10, 20);
print "p2: ($p2[0], $p2[1])\n";

# A C++ std::pair<double,double> comes back as an array reference of two.
my $p3 = make_dpair(1.25, 2.75);
print "make_dpair(1.25, 2.75): ($p3->[0], $p3->[1]), sum_dpair = ", sum_dpair($p3), "\n";
