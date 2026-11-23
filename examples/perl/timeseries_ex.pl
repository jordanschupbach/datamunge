# CMD: just TARGET=timeseries_ex run-perl
# Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $pi = 4 * atan2(1, 1);
my $rng = DotNetRandom->new(42);
sub seed { $rng = DotNetRandom->new($_[0]) }
sub uniform { return 1 - $rng->next_double() }  # (0, 1]: avoids log(0) in the Box-Muller transform.
sub gauss { return sqrt(-2 * log(uniform())) * cos(2 * $pi * uniform()) }
sub joined { return join(", ", map { fmt($_) } @{ $_[0] }) }

# An options object with the given fields set (the C++ structs have no keyword constructor).
sub options {
    my ($class, %fields) = @_;
    my $options = $class->new();
    $options->$_($fields{$_}) for sort keys %fields;
    return $options;
}

print "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n";
seed(42);
my @y;
{
    my ($level, $prev_shock) = (100.0, 0.0);
    for my $i (0 .. 149) {
        my $shock = gauss();
        $level += 0.3 + 0.4 * $prev_shock + $shock;  # integrated ARMA(0,1) increments with drift
        push @y, $level;
        $prev_shock = $shock;
    }
}

my $model = Datamunge::Arima->new(\@y, options('Datamunge::ArimaOptions',
    p => 1, d => 1, q => 1, de_population_size => 80, de_max_generations => 400));

print "AR coefficient: ", fmt($model->ar_coefficients()->[0]), "\n";
print "MA coefficient: ", fmt($model->ma_coefficients()->[0]), "\n";
print "sigma^2: ", fmt($model->sigma2()), ", AIC: ", fmt($model->aic()), ", BIC: ", fmt($model->bic()), "\n";

my ($forecast_values, $std_errors) = @{ $model->forecast_with_intervals(6) };
print "6-step forecast: ", joined($forecast_values), "\n";
print "forecast std. errors: ", joined($std_errors), "\n";

print "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n";
seed(7);
my @s;
{
    my $prev = 0.0;
    for my $i (0 .. 119) {
        $prev = 0.5 * $prev + gauss();
        push @s, 20 + 0.2 * $i + 5 * sin(2 * $pi * $i / 12) + $prev;
    }
}

my $model2 = Datamunge::Arima->new(\@s, options('Datamunge::ArimaOptions',
    p => 1,
    seasonal_p => 1,
    seasonal_d => 1,
    seasonal_period => 12,
    de_population_size => 100,
    de_max_generations => 500,
));
print "AR coefficient: ", fmt($model2->ar_coefficients()->[0]), ", seasonal AR coefficient: ", fmt($model2->seasonal_ar_coefficients()->[0]), "\n";
print "12-step forecast: ", joined($model2->forecast(12)), "\n";

print "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n";
my @seasonal_shape = (0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6);
seed(11);
my @y2 = map { (100 + 2 * $_) * $seasonal_shape[$_ % 12] + gauss() * 3 } 0 .. 47;

my $es_model = Datamunge::ExponentialSmoothing->new(\@y2, options('Datamunge::ExponentialSmoothingOptions',
    trend => Datamunge::TrendType::ADDITIVE,
    seasonal => Datamunge::SeasonalType::MULTIPLICATIVE,
    seasonal_period => 12,
    de_population_size => 60,
    de_max_generations => 300,
));
print "alpha=", fmt($es_model->alpha()), " beta=", fmt($es_model->beta()), " gamma=", fmt($es_model->gamma()), "\n";
print "sigma^2: ", fmt($es_model->sigma2()), ", AIC: ", fmt($es_model->aic()), "\n";
print "12-month forecast: ", joined($es_model->forecast(12)), "\n";

print "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n";
my $flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

my $ses = Datamunge::ExponentialSmoothing->new($flat, Datamunge::ExponentialSmoothingOptions->new());
print "SES alpha= ", fmt($ses->alpha()), "\n";
print "SES 5-step forecast: ", joined($ses->forecast(5)), "\n";

my $holt = Datamunge::ExponentialSmoothing->new($flat, options('Datamunge::ExponentialSmoothingOptions', trend => Datamunge::TrendType::ADDITIVE));
print "Holt alpha=", fmt($holt->alpha()), " beta=", fmt($holt->beta()), "\n";
print "Holt 5-step forecast: ", joined($holt->forecast(5)), "\n";

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
