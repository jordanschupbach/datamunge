% CMD: just TARGET=hypothesis_tests_ex run-octave
% Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
1;

% Shortest round-trip decimal text for x (like C#/Julia default double printing).
function s = num (x)
  if x == 0
    s = '0.0';
    return;
  end
  for d = 1:17
    s = sprintf ('%.*e', d - 1, x);
    if str2double (s) == x
      break;
    end
  end
  epos = strfind (s, 'e');
  e = str2double (s(epos+1:end));
  if e >= -5 && e < 16
    s = sprintf ('%.*f', max (1, d - 1 - e), x);
  else
    s = sprintf ('%se%d', s(1:epos-1), e);
  end
end

function print_result (label, r)
  switch r.alternative
    case datamunge.Alternative.Less
      alternative = 'less';
    case datamunge.Alternative.Greater
      alternative = 'greater';
    otherwise
      alternative = 'two.sided';
  end
  printf ('%s: statistic=%s', label, num (r.statistic));
  if r.parameter1 != 0
    printf (', df1=%s', num (r.parameter1));
  end
  if r.parameter2 != 0
    printf (', df2=%s', num (r.parameter2));
  end
  printf (', p=%s (%s)', num (r.p_value), alternative);
  if r.has_conf_int
    printf (', CI=[%s, %s]', num (r.conf_int_lower), num (r.conf_int_upper));
  end
  printf (' -- %s\n', r.method);
end

function values = column_for_species (df, column, species)
  values = df.pull_numeric (column)(strcmp (df.pull_string ('Species'), species));
end

iris = datamunge.Data.iris ();

setosa_petal = column_for_species (iris, 'Petal.Length', 'setosa');
versicolor_petal = column_for_species (iris, 'Petal.Length', 'versicolor');
virginica_petal = column_for_species (iris, 'Petal.Length', 'virginica');

printf ('=================== t-tests: petal length, setosa vs. versicolor ===================\n');
print_result ('Welch two-sample t-test', datamunge.t_test_two_sample (setosa_petal, versicolor_petal));
print_result ('Wilcoxon rank-sum test', datamunge.wilcoxon_rank_sum_test (setosa_petal, versicolor_petal));

printf ('\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n');
all_petal = [setosa_petal, versicolor_petal, virginica_petal];
sizes = [numel(setosa_petal), numel(versicolor_petal), numel(virginica_petal)];
print_result ('One-way ANOVA', datamunge.one_way_anova (all_petal, sizes));
print_result ('Kruskal-Wallis', datamunge.kruskal_wallis_test (all_petal, sizes));

printf ('\n=================== Correlation: sepal length vs. petal length ===================\n');
sepal_length = iris.pull_numeric ('Sepal.Length');
petal_length = iris.pull_numeric ('Petal.Length');
print_result ('Pearson correlation', datamunge.pearson_correlation_test (sepal_length, petal_length));
print_result ('Spearman correlation', datamunge.spearman_correlation_test (sepal_length, petal_length));

printf ('\n=================== F-test: petal length variance, setosa vs. virginica ===================\n');
print_result ('F test', datamunge.f_test_variance (setosa_petal, virginica_petal));

printf ('\n=================== Normality: is sepal length normally distributed within setosa? ===================\n');
setosa_sepal = column_for_species (iris, 'Sepal.Length', 'setosa');
print_result ('Shapiro-Francia', datamunge.shapiro_francia_test (setosa_sepal));
print_result ('KS vs. fitted normal', datamunge.ks_test_one_sample_normal (setosa_sepal, 5.006, 0.3525));

printf ('\n=================== Chi-squared / Fisher: is petal length "long" independent of species? ===================\n');
sorted_petal = sort (all_petal);
median_all = sorted_petal(floor (numel (all_petal) / 2) + 1);
count_long = @(v) sum (v > median_all);
setosa_long = count_long (setosa_petal);
setosa_short = numel (setosa_petal) - setosa_long;
versicolor_long = count_long (versicolor_petal);
versicolor_short = numel (versicolor_petal) - versicolor_long;
printf ('table: setosa=[%d,%d] versicolor=[%d,%d]\n', setosa_long, setosa_short, versicolor_long, versicolor_short);
table = [setosa_long, setosa_short, versicolor_long, versicolor_short];
print_result ('Chi-squared independence', datamunge.chi_squared_test_independence (table, 2, 2));
print_result ('Fisher''s exact test', datamunge.fisher_exact_test_2x2 (setosa_long, setosa_short, versicolor_long, versicolor_short));

printf ('\n=================== Proportion / binomial: fraction of "long" petals overall ===================\n');
long_count = count_long (all_petal);
n = numel (all_petal);
print_result ('One-sample proportion test (vs 0.5)', datamunge.proportion_test_one_sample (long_count, n, 0.5));
print_result ('Exact binomial test (vs 0.5)', datamunge.binomial_test (long_count, n, 0.5));
