# CMD: just TARGET=datamunge_ex run-perl
# Smoke test of the basic bindings: vectors, pairs and a native Callback object.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

hello();

my $v = make_dvector(1.0, 2.0, 3.0);
print "sum_dvector:\t", sum_dvector($v), "\n";

my $p = make_dpair(1.25, 2.75);
print "sum_dpair:\t", sum_dpair($p), "\n";

my $cb = Datamunge::Callback->new();
print "call_with_callback(3.0):\t", call_with_callback(3.0, $cb), "\n";
my $v2 = map_dvector_with_callback(make_dvector(1.0, 2.0, 3.0), $cb);
print "sum_dvector(map_dvector_with_callback(1,2,3)):\t", sum_dvector($v2), "\n";
