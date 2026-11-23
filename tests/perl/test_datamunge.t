# Smoke tests of the Rosetta-generated Perl binding (just test-perl: prove with
# PERL5LIB=build/perl/prefix/lib/perl5).
use strict;
use warnings;

use Test::More;
use Scalar::Util qw(blessed);

use Datamunge qw(:all);

ok(1, 'loaded Datamunge');
hello();

# std::vector<double> and std::pair<double,double> round trips.
my $v = make_dvector(1.0, 2.0, 3.0);
is_deeply($v, [1.0, 2.0, 3.0], 'std::vector<double> is an array reference');
is(sum_dvector($v), 6.0, 'sum_dvector');
is(sum_dvector([0.5, 1.5]), 2.0, 'array references convert to std::vector');
my $p = make_dpair(1.25, 2.75);
is_deeply($p, [1.25, 2.75], 'std::pair is an array reference of two');
is(sum_dpair($p), 4.0, 'sum_dpair');

{
    package TimesTwo;
    use parent -norequire, 'Datamunge::Callback';
    sub call { my ($self, $x) = @_; return 2 * $x }
}
{
    package Failing;
    use parent -norequire, 'Datamunge::Callback';
    sub call { die { from => 'perl' } }
}

# The C++ Callback class, and a Perl subclass overriding its virtual call() (a director).
my $native = Datamunge::Callback->new();
is(call_with_callback(3.0, $native), 3.0, 'native Callback');
is(sum_dvector(map_dvector_with_callback([1.0, 2.0, 3.0], $native)), 6.0, 'native map');
my $doubler = TimesTwo->new();
is(call_with_callback(3.0, $doubler), 6.0, 'the Perl override runs');
is_deeply(map_dvector_with_callback([1.0, 2.0, 3.0], $doubler), [2.0, 4.0, 6.0], 'override from C++ loops');

# A Perl error inside an override crosses C++ unchanged.
eval { call_with_callback(1.0, Failing->new()) };
is(ref $@ eq 'HASH' && $@->{from}, 'perl', 'the Perl error propagates');

# C++ exceptions die with Datamunge::Error objects.
eval { runif(1.0, 0.0, 3) };
ok(blessed($@) && $@->isa('Datamunge::Error::InvalidArgument') && $@->kind eq 'invalid-argument', 'C++ exceptions die with Datamunge::Error');

# Classes, methods, the policy package Datamunge::Data and the POD documentation.
my $frame = Datamunge::DataFrame->new();
$frame->add_numeric_column('x', [1.0, 2.0, 3.0]);
is($frame->nrows, 3, 'DataFrame');
is_deeply(seql(0.0, 1.0, 5), [0.0, 0.25, 0.5, 0.75, 1.0], 'seql');
is(Datamunge::Data::iris()->nrows, 150, 'the Data package');
open my $pm, '<', $INC{'Datamunge.pm'} or die "Datamunge.pm: $!";
my $pod = do { local $/; <$pm> };
like($pod, qr/evenly spaced/, 'POD documentation');

done_testing();
