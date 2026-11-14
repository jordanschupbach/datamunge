(* CMD: just TARGET=hypothesis_tests_ex run-ocaml *)
(* Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris. *)
open Datamunge

let print_result label r =
  let module R = HypothesisTestResult in
  let alternative =
    match R.alternative r with
    | Alternative.TwoSided -> "two.sided"
    | Alternative.Less -> "less"
    | Alternative.Greater -> "greater"
  in
  let line = Buffer.create 128 in
  Printf.bprintf line "%s: statistic=%g" label (R.statistic r);
  if R.parameter1 r <> 0.0 then Printf.bprintf line ", df1=%g" (R.parameter1 r);
  if R.parameter2 r <> 0.0 then Printf.bprintf line ", df2=%g" (R.parameter2 r);
  Printf.bprintf line ", p=%g (%s)" (R.p_value r) alternative;
  if R.has_conf_int r then
    Printf.bprintf line ", CI=[%g, %g]" (R.conf_int_lower r) (R.conf_int_upper r);
  Printf.printf "%s -- %s\n" (Buffer.contents line) (R.method_ r);
  R.close r

(* Values of a numeric column, optionally only for rows of one species. *)
let column ?species df name =
  List.init (DataFrame.nrows df) Fun.id
  |> List.filter (fun i ->
         match species with
         | None -> true
         | Some s -> DataFrame.string_at df "Species" i = s)
  |> List.map (DataFrame.numeric_at df name)
  |> Array.of_list

(* Number of values strictly greater than threshold. *)
let count_above values threshold =
  Array.fold_left (fun n x -> if x > threshold then n + 1 else n) 0 values

let () =
  let iris = Data.iris () in

  let setosa_petal = column ~species:"setosa" iris "Petal.Length" in
  let versicolor_petal = column ~species:"versicolor" iris "Petal.Length" in
  let virginica_petal = column ~species:"virginica" iris "Petal.Length" in

  print_endline
    "=================== t-tests: petal length, setosa vs. versicolor ===================";
  print_result "Welch two-sample t-test" (t_test_two_sample setosa_petal versicolor_petal);
  print_result "Wilcoxon rank-sum test" (wilcoxon_rank_sum_test setosa_petal versicolor_petal);

  print_endline
    "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================";
  let groups = [ setosa_petal; versicolor_petal; virginica_petal ] in
  let all_petal = Array.concat groups in
  let sizes = List.map Array.length groups in
  print_result "One-way ANOVA" (one_way_anova all_petal sizes);
  print_result "Kruskal-Wallis" (kruskal_wallis_test all_petal sizes);

  print_endline
    "\n=================== Correlation: sepal length vs. petal length ===================";
  let sepal_length = column iris "Sepal.Length" in
  let petal_length = column iris "Petal.Length" in
  print_result "Pearson correlation" (pearson_correlation_test sepal_length petal_length);
  print_result "Spearman correlation" (spearman_correlation_test sepal_length petal_length);

  print_endline
    "\n=================== F-test: petal length variance, setosa vs. virginica ===================";
  print_result "F test" (f_test_variance setosa_petal virginica_petal);

  print_endline
    "\n\
     =================== Normality: is sepal length normally distributed within setosa? \
     ===================";
  let setosa_sepal = column ~species:"setosa" iris "Sepal.Length" in
  print_result "Shapiro-Francia" (shapiro_francia_test setosa_sepal);
  print_result "KS vs. fitted normal"
    (ks_test_one_sample_normal ~mean:5.006 ~sd:0.3525 setosa_sepal);

  print_endline
    "\n\
     =================== Chi-squared / Fisher: is petal length \"long\" independent of species? \
     ===================";
  let sorted = Array.copy all_petal in
  Array.sort compare sorted;
  let median_all = sorted.(Array.length sorted / 2) in
  let setosa_long = count_above setosa_petal median_all in
  let setosa_short = Array.length setosa_petal - setosa_long in
  let versicolor_long = count_above versicolor_petal median_all in
  let versicolor_short = Array.length versicolor_petal - versicolor_long in
  Printf.printf "table: setosa=[%d,%d] versicolor=[%d,%d]\n" setosa_long setosa_short
    versicolor_long versicolor_short;
  let table =
    Array.map float_of_int [| setosa_long; setosa_short; versicolor_long; versicolor_short |]
  in
  print_result "Chi-squared independence" (chi_squared_test_independence table 2 2);
  print_result "Fisher's exact test"
    (fisher_exact_test_2x2 setosa_long setosa_short versicolor_long versicolor_short);

  print_endline
    "\n\
     =================== Proportion / binomial: fraction of \"long\" petals overall \
     ===================";
  let long_count = count_above all_petal median_all in
  let n = Array.length all_petal in
  print_result "One-sample proportion test (vs 0.5)"
    (proportion_test_one_sample ~p:0.5 long_count n);
  print_result "Exact binomial test (vs 0.5)" (binomial_test ~p:0.5 long_count n)
