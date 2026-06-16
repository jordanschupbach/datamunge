use strict;
use warnings;

use Test::More;

use Octra;

ok(1, "Loaded Octra module");

Octra::hello();

my $dp = Octra::DPair->new(1.0, 2.0);
is($dp->swig_first_get(), 1.0, "DPair first");
is($dp->swig_second_get(), 2.0, "DPair second");

{
  package TimesTwo;
  our @ISA = ('Octra::Callback');

  sub call {
    my ($self, $x) = @_;
    return $x * 2.0;
  }
}

my $cb = TimesTwo->new();
is(Octra::call_with_callback(3.0, $cb), 6.0, "call_with_callback uses Perl callback");

my $v = Octra::make_dvector(1.0, 2.0, 3.0);
my $out = Octra::map_dvector_with_callback($v, $cb);
is(Octra::sum_dvector($out), 12.0, "map_dvector_with_callback uses Perl callback");

done_testing();
