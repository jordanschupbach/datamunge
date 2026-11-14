# CMD: just TARGET=hypothesis_tests_ex run-tcl
# Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
package require datamunge

namespace import datamunge::*

proc print_result {label r} {
    set alternative [dict get {TwoSided two.sided Less less Greater greater} [$r alternative]]
    set line "$label: statistic=[$r statistic]"
    if {[$r parameter1] != 0} {
        append line ", df1=[$r parameter1]"
    }
    if {[$r parameter2] != 0} {
        append line ", df2=[$r parameter2]"
    }
    append line ", p=[$r p_value] ($alternative)"
    if {[$r has_conf_int]} {
        append line ", CI=\[[$r conf_int_lower], [$r conf_int_upper]\]"
    }
    puts "$line -- [$r method]"
    $r destroy
}

# Values of a numeric column, optionally only for rows of one species.
proc column {df name {species ""}} {
    set values {}
    for {set i 0} {$i < [$df nrows]} {incr i} {
        if {$species eq "" || [$df string_at Species $i] eq $species} {
            lappend values [$df numeric_at $name $i]
        }
    }
    return $values
}

# Number of values strictly greater than threshold.
proc count_above {values threshold} {
    set count 0
    foreach x $values {
        if {$x > $threshold} {
            incr count
        }
    }
    return $count
}

set iris [datamunge::Data::iris]

set setosa_petal [column $iris Petal.Length setosa]
set versicolor_petal [column $iris Petal.Length versicolor]
set virginica_petal [column $iris Petal.Length virginica]

puts "=================== t-tests: petal length, setosa vs. versicolor ==================="
print_result "Welch two-sample t-test" [t_test_two_sample $setosa_petal $versicolor_petal]
print_result "Wilcoxon rank-sum test" [wilcoxon_rank_sum_test $setosa_petal $versicolor_petal]

puts "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================="
set all_petal [concat $setosa_petal $versicolor_petal $virginica_petal]
set sizes [lmap v [list $setosa_petal $versicolor_petal $virginica_petal] {llength $v}]
print_result "One-way ANOVA" [one_way_anova $all_petal $sizes]
print_result "Kruskal-Wallis" [kruskal_wallis_test $all_petal $sizes]

puts "\n=================== Correlation: sepal length vs. petal length ==================="
set sepal_length [column $iris Sepal.Length]
set petal_length [column $iris Petal.Length]
print_result "Pearson correlation" [pearson_correlation_test $sepal_length $petal_length]
print_result "Spearman correlation" [spearman_correlation_test $sepal_length $petal_length]

puts "\n=================== F-test: petal length variance, setosa vs. virginica ==================="
print_result "F test" [f_test_variance $setosa_petal $virginica_petal]

puts "\n=================== Normality: is sepal length normally distributed within setosa? ==================="
set setosa_sepal [column $iris Sepal.Length setosa]
print_result "Shapiro-Francia" [shapiro_francia_test $setosa_sepal]
print_result "KS vs. fitted normal" [ks_test_one_sample_normal $setosa_sepal 5.006 0.3525]

puts "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ==================="
set median_all [lindex [lsort -real $all_petal] [expr {[llength $all_petal] / 2}]]
set setosa_long [count_above $setosa_petal $median_all]
set setosa_short [expr {[llength $setosa_petal] - $setosa_long}]
set versicolor_long [count_above $versicolor_petal $median_all]
set versicolor_short [expr {[llength $versicolor_petal] - $versicolor_long}]
puts "table: setosa=\[$setosa_long,$setosa_short\] versicolor=\[$versicolor_long,$versicolor_short\]"
set table [lmap c [list $setosa_long $setosa_short $versicolor_long $versicolor_short] {expr {double($c)}}]
print_result "Chi-squared independence" [chi_squared_test_independence $table 2 2]
print_result "Fisher's exact test" \
    [fisher_exact_test_2x2 $setosa_long $setosa_short $versicolor_long $versicolor_short]

puts "\n=================== Proportion / binomial: fraction of \"long\" petals overall ==================="
set long_count [count_above $all_petal $median_all]
set n [llength $all_petal]
print_result "One-sample proportion test (vs 0.5)" [proportion_test_one_sample $long_count $n 0.5]
print_result "Exact binomial test (vs 0.5)" [binomial_test $long_count $n 0.5]
