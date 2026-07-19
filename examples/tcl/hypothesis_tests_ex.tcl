package require Datamunge 0.0.1

proc print_result {label r} {
  set line "$label: statistic=[datamunge::HypothesisTestResult_statistic_get $r]"
  if {[datamunge::HypothesisTestResult_parameter1_get $r] != 0.0} {
    append line ", df1=[datamunge::HypothesisTestResult_parameter1_get $r]"
  }
  if {[datamunge::HypothesisTestResult_parameter2_get $r] != 0.0} {
    append line ", df2=[datamunge::HypothesisTestResult_parameter2_get $r]"
  }
  set alt_names {two.sided less greater}
  append line ", p=[datamunge::HypothesisTestResult_p_value_get $r] ([lindex $alt_names [datamunge::HypothesisTestResult_alternative_get $r]])"
  if {[datamunge::HypothesisTestResult_has_conf_int_get $r]} {
    append line ", CI=\[[datamunge::HypothesisTestResult_conf_int_lower_get $r], [datamunge::HypothesisTestResult_conf_int_upper_get $r]\]"
  }
  append line " -- [datamunge::HypothesisTestResult_method_get $r]"
  puts $line
}

proc column_for_species {df column species} {
  set out {}
  set n [datamunge::DataFrame_nrows $df]
  for {set i 0} {$i < $n} {incr i} {
    if {[datamunge::DataFrame_string_at $df "Species" $i] eq $species} {
      lappend out [datamunge::DataFrame_numeric_at $df $column $i]
    }
  }
  return $out
}

proc szv {values} {
  set v [datamunge::new_SizeVector]
  foreach x $values { datamunge::SizeVector_push $v $x }
  return $v
}

set iris [datamunge::DataFrame_iris]

set setosa_petal [column_for_species $iris "Petal.Length" "setosa"]
set versicolor_petal [column_for_species $iris "Petal.Length" "versicolor"]
set virginica_petal [column_for_species $iris "Petal.Length" "virginica"]

puts "=================== t-tests: petal length, setosa vs. versicolor ==================="
print_result "Welch two-sample t-test" [datamunge::t_test_two_sample $setosa_petal $versicolor_petal]
print_result "Wilcoxon rank-sum test" [datamunge::wilcoxon_rank_sum_test $setosa_petal $versicolor_petal]

puts "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================="
set all_petal [concat $setosa_petal $versicolor_petal $virginica_petal]
set sizes [szv [list [llength $setosa_petal] [llength $versicolor_petal] [llength $virginica_petal]]]
print_result "One-way ANOVA" [datamunge::one_way_anova $all_petal $sizes]
print_result "Kruskal-Wallis" [datamunge::kruskal_wallis_test $all_petal $sizes]

puts "\n=================== Correlation: sepal length vs. petal length ==================="
set n [datamunge::DataFrame_nrows $iris]
set sepal_length {}
set petal_length {}
for {set i 0} {$i < $n} {incr i} {
  lappend sepal_length [datamunge::DataFrame_numeric_at $iris "Sepal.Length" $i]
  lappend petal_length [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
}
print_result "Pearson correlation" [datamunge::pearson_correlation_test $sepal_length $petal_length]
print_result "Spearman correlation" [datamunge::spearman_correlation_test $sepal_length $petal_length]

puts "\n=================== F-test: petal length variance, setosa vs. virginica ==================="
print_result "F test" [datamunge::f_test_variance $setosa_petal $virginica_petal]

puts "\n=================== Normality: is sepal length normally distributed within setosa? ==================="
set setosa_sepal [column_for_species $iris "Sepal.Length" "setosa"]
print_result "Shapiro-Francia" [datamunge::shapiro_francia_test $setosa_sepal]
print_result "KS vs. fitted normal" [datamunge::ks_test_one_sample_normal $setosa_sepal 5.006 0.3525]

puts "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ==================="
set sorted_all [lsort -real $all_petal]
set median_all [lindex $sorted_all [expr {int([llength $all_petal] / 2)}]]
set setosa_long 0
foreach x $setosa_petal { if {$x > $median_all} { incr setosa_long } }
set setosa_short [expr {[llength $setosa_petal] - $setosa_long}]
set versicolor_long 0
foreach x $versicolor_petal { if {$x > $median_all} { incr versicolor_long } }
set versicolor_short [expr {[llength $versicolor_petal] - $versicolor_long}]
puts "table: setosa=\[$setosa_long,$setosa_short\] versicolor=\[$versicolor_long,$versicolor_short\]"
set table_vec [list $setosa_long $setosa_short $versicolor_long $versicolor_short]
print_result "Chi-squared independence" [datamunge::chi_squared_test_independence $table_vec 2 2]
print_result "Fisher's exact test" [datamunge::fisher_exact_test_2x2 $setosa_long $setosa_short $versicolor_long $versicolor_short]

puts "\n=================== Proportion / binomial: fraction of \"long\" petals overall ==================="
set long_count 0
foreach x $all_petal { if {$x > $median_all} { incr long_count } }
print_result "One-sample proportion test (vs 0.5)" [datamunge::proportion_test_one_sample $long_count [llength $all_petal] 0.5]
print_result "Exact binomial test (vs 0.5)" [datamunge::binomial_test $long_count [llength $all_petal] 0.5]
