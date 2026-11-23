# CMD: just TARGET=hypothesis_tests_ex run-perl
# Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

sub print_result {
    my ($label, $r) = @_;
    my $alternative = $r->alternative == Datamunge::Alternative::LESS ? "less"
        : $r->alternative == Datamunge::Alternative::GREATER ? "greater"
        : "two.sided";
    print "$label: statistic=", fmt($r->statistic);
    print ", df1=", fmt($r->parameter1) if $r->parameter1 != 0;
    print ", df2=", fmt($r->parameter2) if $r->parameter2 != 0;
    print ", p=", fmt($r->p_value), " ($alternative)";
    print ", CI=[", fmt($r->conf_int_lower), ", ", fmt($r->conf_int_upper), "]" if $r->has_conf_int;
    print " -- ", $r->method, "\n";
}

sub column_for_species {
    my ($df, $column, $species) = @_;
    return [map { $df->numeric_at($column, $_) } grep { $df->string_at("Species", $_) eq $species } 0 .. $df->nrows() - 1];
}

sub column {
    my ($df, $column) = @_;
    return [map { $df->numeric_at($column, $_) } 0 .. $df->nrows() - 1];
}

my $iris = Datamunge::Data::iris();

my $setosa_petal = column_for_species($iris, "Petal.Length", "setosa");
my $versicolor_petal = column_for_species($iris, "Petal.Length", "versicolor");
my $virginica_petal = column_for_species($iris, "Petal.Length", "virginica");

print "=================== t-tests: petal length, setosa vs. versicolor ===================\n";
print_result("Welch two-sample t-test", t_test_two_sample($setosa_petal, $versicolor_petal));
print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test($setosa_petal, $versicolor_petal));

print "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n";
my @all_petal = (@$setosa_petal, @$versicolor_petal, @$virginica_petal);
my $sizes = [scalar @$setosa_petal, scalar @$versicolor_petal, scalar @$virginica_petal];
print_result("One-way ANOVA", one_way_anova(\@all_petal, $sizes));
print_result("Kruskal-Wallis", kruskal_wallis_test(\@all_petal, $sizes));

print "\n=================== Correlation: sepal length vs. petal length ===================\n";
my $sepal_length = column($iris, "Sepal.Length");
my $petal_length = column($iris, "Petal.Length");
print_result("Pearson correlation", pearson_correlation_test($sepal_length, $petal_length));
print_result("Spearman correlation", spearman_correlation_test($sepal_length, $petal_length));

print "\n=================== F-test: petal length variance, setosa vs. virginica ===================\n";
print_result("F test", f_test_variance($setosa_petal, $virginica_petal));

print "\n=================== Normality: is sepal length normally distributed within setosa? ===================\n";
my $setosa_sepal = column_for_species($iris, "Sepal.Length", "setosa");
print_result("Shapiro-Francia", shapiro_francia_test($setosa_sepal));
print_result("KS vs. fitted normal", ks_test_one_sample_normal($setosa_sepal, 5.006, 0.3525));

print "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n";
my $median_all = (sort { $a <=> $b } @all_petal)[int(@all_petal / 2)];
sub count_long { my ($v) = @_; return scalar grep { $_ > $median_all } @$v }
{
    my $setosa_long = count_long($setosa_petal);
    my $setosa_short = @$setosa_petal - $setosa_long;
    my $versicolor_long = count_long($versicolor_petal);
    my $versicolor_short = @$versicolor_petal - $versicolor_long;
    print "table: setosa=[$setosa_long,$setosa_short] versicolor=[$versicolor_long,$versicolor_short]\n";
    my $table = [$setosa_long, $setosa_short, $versicolor_long, $versicolor_short];
    print_result("Chi-squared independence", chi_squared_test_independence($table, 2, 2));
    print_result("Fisher's exact test", fisher_exact_test_2x2($setosa_long, $setosa_short, $versicolor_long, $versicolor_short));
}

print "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n";
{
    my $long_count = count_long(\@all_petal);
    my $n = @all_petal;
    print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample($long_count, $n, 0.5));
    print_result("Exact binomial test (vs 0.5)", binomial_test($long_count, $n, 0.5));
}

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
