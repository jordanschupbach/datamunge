use strict;
use warnings;

use Test::More;

use Datamunge;

ok(1, "Loaded Datamunge module");

Datamunge::hello();

my $dp = Datamunge::DPair->new(1.0, 2.0);
is($dp->swig_first_get(), 1.0, "DPair first");
is($dp->swig_second_get(), 2.0, "DPair second");

{
  package TimesTwo;
  our @ISA = ('Datamunge::Callback');

  sub call {
    my ($self, $x) = @_;
    return $x * 2.0;
  }
}

my $cb = TimesTwo->new();
is(Datamunge::call_with_callback(3.0, $cb), 6.0, "call_with_callback uses Perl callback");

my $v = Datamunge::make_dvector(1.0, 2.0, 3.0);
my $out = Datamunge::map_dvector_with_callback($v, $cb);
is(Datamunge::sum_dvector($out), 12.0, "map_dvector_with_callback uses Perl callback");

done_testing();
