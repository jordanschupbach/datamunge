use strict;
use warnings;

use Datamunge;

my %ALT_NAMES = (
  $Datamunge::Alternative_Less => "less",
  $Datamunge::Alternative_Greater => "greater",
  $Datamunge::Alternative_TwoSided => "two.sided",
);

sub print_result {
  my ($label, $r) = @_;
  my $line = "$label: statistic=" . $r->swig_statistic_get();
  $line .= ", df1=" . $r->swig_parameter1_get() if $r->swig_parameter1_get() != 0.0;
  $line .= ", df2=" . $r->swig_parameter2_get() if $r->swig_parameter2_get() != 0.0;
  $line .= ", p=" . $r->swig_p_value_get() . " (" . $ALT_NAMES{$r->swig_alternative_get()} . ")";
  $line .= ", CI=[" . $r->swig_conf_int_lower_get() . ", " . $r->swig_conf_int_upper_get() . "]" if $r->swig_has_conf_int_get();
  $line .= " -- " . $r->swig_method_get();
  print "$line\n";
}

sub column_for_species {
  my ($df, $column, $species) = @_;
  my @out;
  for (my $i = 0; $i < $df->nrows(); $i++) {
    push @out, $df->numeric_at($column, $i) if $df->string_at("Species", $i) eq $species;
  }
  return \@out;
}

my $iris = Datamunge::DataFrame::iris();

my $setosa_petal = column_for_species($iris, "Petal.Length", "setosa");
my $versicolor_petal = column_for_species($iris, "Petal.Length", "versicolor");
my $virginica_petal = column_for_species($iris, "Petal.Length", "virginica");

print "=================== t-tests: petal length, setosa vs. versicolor ===================\n";
print_result("Welch two-sample t-test", Datamunge::t_test_two_sample($setosa_petal, $versicolor_petal));
print_result("Wilcoxon rank-sum test", Datamunge::wilcoxon_rank_sum_test($setosa_petal, $versicolor_petal));

print "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n";
my @all_petal = (@$setosa_petal, @$versicolor_petal, @$virginica_petal);
my @sizes = (scalar(@$setosa_petal), scalar(@$versicolor_petal), scalar(@$virginica_petal));
print_result("One-way ANOVA", Datamunge::one_way_anova(\@all_petal, \@sizes));
print_result("Kruskal-Wallis", Datamunge::kruskal_wallis_test(\@all_petal, \@sizes));

print "\n=================== Correlation: sepal length vs. petal length ===================\n";
my @sepal_length = map { $iris->numeric_at("Sepal.Length", $_) } (0 .. $iris->nrows() - 1);
my @petal_length = map { $iris->numeric_at("Petal.Length", $_) } (0 .. $iris->nrows() - 1);
print_result("Pearson correlation", Datamunge::pearson_correlation_test(\@sepal_length, \@petal_length));
print_result("Spearman correlation", Datamunge::spearman_correlation_test(\@sepal_length, \@petal_length));

print "\n=================== F-test: petal length variance, setosa vs. virginica ===================\n";
print_result("F test", Datamunge::f_test_variance($setosa_petal, $virginica_petal));

print "\n=================== Normality: is sepal length normally distributed within setosa? ===================\n";
my $setosa_sepal = column_for_species($iris, "Sepal.Length", "setosa");
print_result("Shapiro-Francia", Datamunge::shapiro_francia_test($setosa_sepal));
print_result("KS vs. fitted normal", Datamunge::ks_test_one_sample_normal($setosa_sepal, 5.006, 0.3525));

print "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n";
my @sorted_all = sort { $a <=> $b } @all_petal;
my $median_all = $sorted_all[int(scalar(@all_petal) / 2)];
my $setosa_long = scalar(grep { $_ > $median_all } @$setosa_petal);
my $setosa_short = scalar(@$setosa_petal) - $setosa_long;
my $versicolor_long = scalar(grep { $_ > $median_all } @$versicolor_petal);
my $versicolor_short = scalar(@$versicolor_petal) - $versicolor_long;
print "table: setosa=[$setosa_long,$setosa_short] versicolor=[$versicolor_long,$versicolor_short]\n";
my @table = ($setosa_long * 1.0, $setosa_short * 1.0, $versicolor_long * 1.0, $versicolor_short * 1.0);
print_result("Chi-squared independence", Datamunge::chi_squared_test_independence(\@table, 2, 2));
print_result("Fisher's exact test", Datamunge::fisher_exact_test_2x2($setosa_long, $setosa_short, $versicolor_long, $versicolor_short));

print "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n";
my $long_count = scalar(grep { $_ > $median_all } @all_petal);
print_result("One-sample proportion test (vs 0.5)", Datamunge::proportion_test_one_sample($long_count, scalar(@all_petal), 0.5));
print_result("Exact binomial test (vs 0.5)", Datamunge::binomial_test($long_count, scalar(@all_petal), 0.5));
