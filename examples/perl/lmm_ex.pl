# CMD: just TARGET=lmm_ex run-perl
# Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $pi = 4 * atan2(1, 1);
my $rng = DotNetRandom->new(2024);
sub uniform { return 1 - $rng->next_double() }  # (0, 1]: avoids log(0) in the Box-Muller transform.
sub gauss { return sqrt(-2 * log(uniform())) * cos(2 * $pi * uniform()) }

print "=================== Random intercept on a real dataset (penguins) ===================\n";
Datamunge::Lmm->new(Datamunge::Data::penguins(), "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")->print_summary();

print "\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n";
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
my $n_schools = 30;
my @school_intercept = map { gauss() * 6 } 1 .. $n_schools;
my @school_slope = map { gauss() * 1.2 } 1 .. $n_schools;
my $true_intercept = 60.0;
my $true_slope = 3.0;

my (@school, @study_hours, @score);
for my $s (0 .. $n_schools - 1) {
    my $n_students = 15 + $rng->next_int(0, 21);
    for my $student (0 .. $n_students - 1) {
        my $hours = $rng->next_double() * 10;
        push @school, $s;
        push @study_hours, $hours;
        push @score, $true_intercept + $school_intercept[$s] + ($true_slope + $school_slope[$s]) * $hours + gauss() * 4;
    }
}
my $df = Datamunge::DataFrame->new();
$df->add_numeric_column("school", \@school);
$df->add_numeric_column("study_hours", \@study_hours);
$df->add_numeric_column("score", \@score);

my $model = Datamunge::Lmm->new($df, "score ~ study_hours + (1 + study_hours | school)");
$model->print_summary();

print "\nTrue generating values: intercept=", fmt($true_intercept), ", slope=", fmt($true_slope), ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n";

print "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n";
my $labels = $model->group_labels();
for my $idx (0 .. 2) {
    my $re = $model->random_effects_for_group($idx);
    print "school $labels->[$idx]: intercept shift=", fmt($re->[0]), ", slope shift=", fmt($re->[1]), "\n";
}

print "\n--- Prediction: population-level vs. school-adjusted ---\n";
my $population = Datamunge::DataFrame->new();
my $school0 = Datamunge::DataFrame->new();
$population->add_numeric_column("study_hours", [5.0]);
$school0->add_numeric_column("study_hours", [5.0]);
$school0->add_numeric_column("school", [0.0]);
print "5 study hours, unseen school:      ", fmt($model->predict($population)->[0]), " (fixed effects only)\n";
print "5 study hours, school 0 (known):    ", fmt($model->predict($school0)->[0]), " (fixed effects + school 0's BLUP)\n";

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
