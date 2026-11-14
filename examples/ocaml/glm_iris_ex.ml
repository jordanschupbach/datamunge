(* CMD: just TARGET=glm_iris_ex run-ocaml *)
(* Generalized linear models on iris: logistic regression (binomial) and Poisson regression. *)
open Datamunge

let column df name rows = Array.of_list (List.map (DataFrame.numeric_at df name) rows)
let mean values = Array.fold_left ( +. ) 0.0 values /. float_of_int (Array.length values)

let () =
  let iris = Data.iris () in
  let rows = List.init (DataFrame.nrows iris) Fun.id in

  (* Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
     the other two on these predictors, which sends logistic regression's coefficients toward
     +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
     binary split for a demo. *)
  let species i = DataFrame.string_at iris "Species" i in
  let kept = List.filter (fun i -> List.mem (species i) [ "versicolor"; "virginica" ]) rows in
  let is_virginica =
    Array.of_list (List.map (fun i -> if species i = "virginica" then 1.0 else 0.0) kept)
  in
  let petal_length = column iris "Petal.Length" kept in
  let petal_width = column iris "Petal.Width" kept in

  let sub = DataFrame.create () in
  DataFrame.add_numeric_column sub "Petal.Length" petal_length;
  DataFrame.add_numeric_column sub "Petal.Width" petal_width;
  DataFrame.add_numeric_column sub "is_virginica" is_virginica;

  print_endline
    "=================== Logistic regression (binomial, logit link) ===================";
  let logit = GLM.create ~family:"binomial" sub "is_virginica ~ Petal.Length + Petal.Width" in
  GLM.print_summary logit;

  let n = Array.length is_virginica in
  let correct = ref 0 in
  Array.iter2
    (fun f y -> if f >= 0.5 = (y >= 0.5) then incr correct)
    (GLM.fitted_values logit) is_virginica;
  Printf.printf "\nResubstitution accuracy at 0.5 threshold: %g%%\n"
    (100.0 *. float_of_int !correct /. float_of_int n);

  GLM.save_diagnostic_plots logit "glm_logistic_iris";
  print_endline
    "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,\
     residuals_vs_leverage}.svg";

  (* Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
     classic sigmoid shape of a fitted logistic regression, with a 95% confidence band. *)
  let grid_n = 100 in
  let width_mean = mean petal_width in
  let pl_min = Array.fold_left Float.min infinity petal_length -. 0.3 in
  let pl_max = Array.fold_left Float.max neg_infinity petal_length +. 0.3 in
  let grid_x =
    Array.init grid_n (fun i ->
        pl_min +. ((pl_max -. pl_min) *. float_of_int i /. float_of_int (grid_n - 1)))
  in
  let grid = DataFrame.create () in
  DataFrame.add_numeric_column grid "Petal.Length" grid_x;
  DataFrame.add_numeric_column grid "Petal.Width" (Array.make grid_n width_mean);
  let curve_frame = GLM.predict_frame ~interval_kind:"confidence" logit grid in
  print_endline "\nPredicted-probability curve (first 5 rows):";
  print_endline (DataFrame.to_string ~max_rows:5 curve_frame);

  (* Poisson regression, for contrast: same IRLS engine, different family/link. *)
  print_endline "\n=================== Poisson regression (log link) ===================";
  let count_data = DataFrame.create () in
  DataFrame.add_numeric_column count_data "Sepal.Width" (column iris "Sepal.Width" rows);
  DataFrame.add_numeric_column count_data "Petal.Length" (column iris "Petal.Length" rows);
  DataFrame.add_numeric_column count_data "count"
    (Array.map (fun x -> Float.floor (x +. 0.5)) (column iris "Sepal.Length" rows));
  let poisson = GLM.create ~family:"poisson" count_data "count ~ Sepal.Width + Petal.Length" in
  GLM.print_summary poisson
