open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let szv values =
  let v = new_SizeVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_int64 (Int64.of_int x)))) values;
  v

let alt_name a =
  match a with
  | C_enum `TwoSided -> "two.sided"
  | C_enum `Less -> "less"
  | C_enum `Greater -> "greater"
  | _ -> "?"

let print_result label r =
  let buf = Buffer.create 64 in
  Buffer.add_string buf (Printf.sprintf "%s: statistic=%g" label (get_float (invoke r "[statistic]" C_void)));
  let p1 = get_float (invoke r "[parameter1]" C_void) in
  if p1 <> 0.0 then Buffer.add_string buf (Printf.sprintf ", df1=%g" p1);
  let p2 = get_float (invoke r "[parameter2]" C_void) in
  if p2 <> 0.0 then Buffer.add_string buf (Printf.sprintf ", df2=%g" p2);
  Buffer.add_string buf (Printf.sprintf ", p=%g (%s)" (get_float (invoke r "[p_value]" C_void)) (alt_name (invoke r "[alternative]" C_void)));
  if get_bool (invoke r "[has_conf_int]" C_void) then
    Buffer.add_string buf (Printf.sprintf ", CI=[%g, %g]" (get_float (invoke r "[conf_int_lower]" C_void)) (get_float (invoke r "[conf_int_upper]" C_void)));
  Buffer.add_string buf (Printf.sprintf " -- %s" (get_string (invoke r "[method]" C_void)));
  print_endline (Buffer.contents buf)

let column_for_species df column species =
  let n = get_int (invoke df "nrows" C_void) in
  let out = ref [] in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    if get_string (invoke df "string_at" (C_list [C_string "Species"; i64])) = species then
      out := get_float (invoke df "numeric_at" (C_list [C_string column; i64])) :: !out
  done;
  List.rev !out

let () =
  let iris = _DataFrame_iris C_void in

  let setosa_petal = column_for_species iris "Petal.Length" "setosa" in
  let versicolor_petal = column_for_species iris "Petal.Length" "versicolor" in
  let virginica_petal = column_for_species iris "Petal.Length" "virginica" in

  print_endline "=================== t-tests: petal length, setosa vs. versicolor ===================";
  print_result "Welch two-sample t-test" (_t_test_two_sample (C_list [dv setosa_petal; dv versicolor_petal]));
  print_result "Wilcoxon rank-sum test" (_wilcoxon_rank_sum_test (C_list [dv setosa_petal; dv versicolor_petal]));

  print_endline "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================";
  let all_petal = setosa_petal @ versicolor_petal @ virginica_petal in
  let sizes = szv [List.length setosa_petal; List.length versicolor_petal; List.length virginica_petal] in
  print_result "One-way ANOVA" (_one_way_anova (C_list [dv all_petal; sizes]));
  print_result "Kruskal-Wallis" (_kruskal_wallis_test (C_list [dv all_petal; sizes]));

  print_endline "\n=================== Correlation: sepal length vs. petal length ===================";
  let n = get_int (invoke iris "nrows" C_void) in
  let sepal_length = List.init n (fun i -> get_float (invoke iris "numeric_at" (C_list [C_string "Sepal.Length"; C_int64 (Int64.of_int i)]))) in
  let petal_length = List.init n (fun i -> get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; C_int64 (Int64.of_int i)]))) in
  print_result "Pearson correlation" (_pearson_correlation_test (C_list [dv sepal_length; dv petal_length]));
  print_result "Spearman correlation" (_spearman_correlation_test (C_list [dv sepal_length; dv petal_length]));

  print_endline "\n=================== F-test: petal length variance, setosa vs. virginica ===================";
  print_result "F test" (_f_test_variance (C_list [dv setosa_petal; dv virginica_petal]));

  print_endline "\n=================== Normality: is sepal length normally distributed within setosa? ===================";
  let setosa_sepal = column_for_species iris "Sepal.Length" "setosa" in
  print_result "Shapiro-Francia" (_shapiro_francia_test (dv setosa_sepal));
  print_result "KS vs. fitted normal" (_ks_test_one_sample_normal (C_list [dv setosa_sepal; C_double 5.006; C_double 0.3525]));

  print_endline "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================";
  let sorted_all = List.sort compare all_petal in
  let median_all = List.nth sorted_all (List.length all_petal / 2) in
  let count_long lst = List.length (List.filter (fun v -> v > median_all) lst) in
  let setosa_long = count_long setosa_petal in
  let setosa_short = List.length setosa_petal - setosa_long in
  let versicolor_long = count_long versicolor_petal in
  let versicolor_short = List.length versicolor_petal - versicolor_long in
  Printf.printf "table: setosa=[%d,%d] versicolor=[%d,%d]\n" setosa_long setosa_short versicolor_long versicolor_short;
  let table_vec = dv [float_of_int setosa_long; float_of_int setosa_short; float_of_int versicolor_long; float_of_int versicolor_short] in
  print_result "Chi-squared independence" (_chi_squared_test_independence (C_list [table_vec; C_int64 2L; C_int64 2L]));
  print_result "Fisher's exact test" (_fisher_exact_test_2x2 (C_list [C_int64 (Int64.of_int setosa_long); C_int64 (Int64.of_int setosa_short); C_int64 (Int64.of_int versicolor_long); C_int64 (Int64.of_int versicolor_short)]));

  print_endline "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================";
  let long_count = count_long all_petal in
  print_result "One-sample proportion test (vs 0.5)" (_proportion_test_one_sample (C_list [C_int64 (Int64.of_int long_count); C_int64 (Int64.of_int (List.length all_petal)); C_double 0.5]));
  print_result "Exact binomial test (vs 0.5)" (_binomial_test (C_list [C_int64 (Int64.of_int long_count); C_int64 (Int64.of_int (List.length all_petal)); C_double 0.5]))
