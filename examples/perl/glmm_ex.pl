# CMD: just TARGET=glmm_ex run-perl
# Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $pi = 4 * atan2(1, 1);
my $rng = DotNetRandom->new(4242);
sub uniform { return $rng->next_double() }
sub gauss {
    my ($u1, $u2) = (1 - uniform(), uniform());  # 1 - u keeps log() away from 0
    return sqrt(-2 * log($u1)) * cos(2 * $pi * $u2);
}

print "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n";
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
{
    my $penguins = Datamunge::Data::penguins();
    my (@is_male, @body_mass, @island);
    for my $i (0 .. $penguins->nrows() - 1) {
        next if grep { $penguins->is_null($_, $i) } "sex", "body_mass_g", "island";
        push @is_male, $penguins->string_at("sex", $i) eq "male" ? 1.0 : 0.0;
        push @body_mass, $penguins->numeric_at("body_mass_g", $i);
        push @island, $penguins->string_at("island", $i);
    }
    my $sex_df = Datamunge::DataFrame->new();
    $sex_df->add_numeric_column("is_male", \@is_male);
    $sex_df->add_numeric_column("body_mass_g", \@body_mass);
    $sex_df->add_string_column("island", \@island);
    Datamunge::Glmm->new($sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")->print_summary();
}

print "\n=================== Poisson mixed model on simulated multi-site count data ===================\n";
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
my $n_stores = 25;
my @store_effect = map { gauss() * 0.4 } 1 .. $n_stores;
my $true_intercept = 2.0;
my $true_slope = 0.3;

my (@store, @promo, @visits);
for my $s (0 .. $n_stores - 1) {
    my $n_days = 15 + $rng->next_int(0, 11);
    for my $day (0 .. $n_days - 1) {
        my $promo_intensity = uniform() * 3;
        my $lam = exp($true_intercept + $store_effect[$s] + $true_slope * $promo_intensity);
        # Knuth's Poisson sampler.
        my ($threshold, $k, $p) = (exp(-$lam), 0, 1.0);
        do {
            $k++;
            $p *= uniform();
        } while ($p > $threshold);
        push @store, $s;
        push @promo, $promo_intensity;
        push @visits, $k - 1;
    }
}
my $df = Datamunge::DataFrame->new();
$df->add_numeric_column("store", \@store);
$df->add_numeric_column("promo", \@promo);
$df->add_numeric_column("visits", \@visits);
my $store_model = Datamunge::Glmm->new($df, "visits ~ promo + (1 | store)", "poisson");

$store_model->print_summary();

print "\nTrue generating values: intercept=", fmt($true_intercept), ", slope=", fmt($true_slope), ", random-intercept SD (log scale)=0.4\n";

print "\n--- BLUPs for a few stores ---\n";
my $labels = $store_model->group_labels();
for my $idx (0 .. 2) {
    print "store $labels->[$idx]: intercept shift=", fmt($store_model->random_effects_for_group($idx)->[0]), "\n";
}

print "\n--- Prediction: population-level vs. store-adjusted ---\n";
my $population = Datamunge::DataFrame->new();
my $store0 = Datamunge::DataFrame->new();
$population->add_numeric_column("promo", [1.5]);
$store0->add_numeric_column("promo", [1.5]);
$store0->add_numeric_column("store", [0.0]);
print "promo=1.5, unseen store:   ", fmt($store_model->predict($population)->[0]), " expected visits (fixed effects only)\n";
print "promo=1.5, store 0 (known): ", fmt($store_model->predict($store0)->[0]), " expected visits (fixed effects + store 0's BLUP)\n";

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

# .NET's seeded System.Random (Knuth's subtractive generator), so the simulated data -- and so
# the output -- matches the C# example exactly.
package DotNetRandom {
    use constant { MBIG => 2147483647, MSEED => 161803398 };

    sub new {
        my ($class, $seed) = @_;
        my @a = (0) x 56;
        my $mj = MSEED - abs($seed);
        $a[55] = $mj;
        my $mk = 1;
        for my $i (1 .. 54) {
            my $ii = (21 * $i) % 55;
            $a[$ii] = $mk;
            $mk = $mj - $mk;
            $mk += MBIG if $mk < 0;
            $mj = $a[$ii];
        }
        for my $k (1 .. 4) {
            for my $i (1 .. 55) {
                $a[$i] -= $a[1 + ($i + 30) % 55];
                $a[$i] += MBIG if $a[$i] < 0;
            }
        }
        return bless { a => \@a, inext => 0, inextp => 21 }, $class;
    }

    sub _sample {
        my $self = shift;
        my $a = $self->{a};
        my $inext = $self->{inext} + 1;
        $inext = 1 if $inext >= 56;
        my $inextp = $self->{inextp} + 1;
        $inextp = 1 if $inextp >= 56;
        my $r = $a->[$inext] - $a->[$inextp];
        $r-- if $r == MBIG;
        $r += MBIG if $r < 0;
        $a->[$inext] = $r;
        @$self{qw(inext inextp)} = ($inext, $inextp);
        return $r;
    }

    # NextDouble(): a number in [0, 1).
    sub next_double { return $_[0]->_sample() * (1.0 / MBIG) }

    # Next(min, max): an integer in [min, max).
    sub next_int { my ($self, $min, $max) = @_; return int($self->next_double() * ($max - $min)) + $min }
}
