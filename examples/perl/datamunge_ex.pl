use strict;
use warnings;

use Datamunge;

Datamunge::hello();

my $dp = Datamunge::DPair->new(1.0, 2.0);
print "DPair: First=" . $dp->swig_first_get() . ", Second=" . $dp->swig_second_get() . "\n";

{
  package TimesTwo;
  our @ISA = ('Datamunge::Callback');

  sub call {
    my ($self, $x) = @_;
    return $x * 2.0;
  }
}

my $cb = TimesTwo->new();
print "call_with_callback(3.0) = ", Datamunge::call_with_callback(3.0, $cb), "\n";
my $v = Datamunge::make_dvector(1.0, 2.0, 3.0);
my $v2 = Datamunge::map_dvector_with_callback($v, $cb);
print "sum_dvector(map_dvector_with_callback(1,2,3)) = ", Datamunge::sum_dvector($v2), "\n";
