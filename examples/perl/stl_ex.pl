use strict;
use warnings;

use Datamunge;

my $n = 100;
my $v = Datamunge::DVector->new($n);

for (my $i = 0; $i < $n; $i++) { $v->set($i, $i * 1.5); }
for (my $i = 0; $i < $n; $i++) { print $v->get($i), "\n"; }

my $v2 = Datamunge::IVector->new($n);
for (my $i = 0; $i < $n; $i++) { $v2->set($i, int($i * 1.5)); }
for (my $i = 0; $i < $n; $i++) { print $v2->get($i), "\n"; }

my $p = Datamunge::IPair->new(3, 4);
print "p: (" . $p->swig_first_get() . ", " . $p->swig_second_get() . ")\n";

my $p2 = Datamunge::DPair->new(10, 20);
print "p2: (" . $p2->swig_first_get() . ", " . $p2->swig_second_get() . ")\n";
