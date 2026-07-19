open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let () =
  let iris = _DataFrame_iris C_void in
  let n = get_int (invoke iris "nrows" C_void) in

  let is_virginica = ref [] and petal_length = ref [] and petal_width = ref [] in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    let species = get_string (invoke iris "string_at" (C_list [C_string "Species"; i64])) in
    if species = "versicolor" || species = "virginica" then begin
      is_virginica := (if species = "virginica" then 1.0 else 0.0) :: !is_virginica;
      petal_length := get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; i64])) :: !petal_length;
      petal_width := get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Width"; i64])) :: !petal_width
    end
  done;
  let is_virginica = List.rev !is_virginica and petal_length = List.rev !petal_length and petal_width = List.rev !petal_width in

  let sub = new_DataFrame C_void in
  ignore (invoke sub "add_numeric_column" (C_list [C_string "Petal.Length"; dv petal_length]));
  ignore (invoke sub "add_numeric_column" (C_list [C_string "Petal.Width"; dv petal_width]));
  ignore (invoke sub "add_numeric_column" (C_list [C_string "is_virginica"; dv is_virginica]));

  print_endline "=================== Logistic regression (binomial, logit link) ===================";
  let logit = new_GLM (C_list [sub; C_string "is_virginica ~ Petal.Length + Petal.Width"; C_string "binomial"]) in
  ignore (invoke logit "print_summary" C_void);

  let fitted = invoke logit "fitted_values" C_void in
  let is_virginica_arr = Array.of_list is_virginica in
  let correct = ref 0 in
  Array.iteri (fun i iv ->
    let f = get_float (invoke fitted "[]" (C_int i)) in
    if (f >= 0.5) = (iv >= 0.5) then incr correct
  ) is_virginica_arr;
  Printf.printf "\nResubstitution accuracy at 0.5 threshold: %g%%\n" (100.0 *. float_of_int !correct /. float_of_int (Array.length is_virginica_arr));

  ignore (invoke logit "save_diagnostic_plots" (C_string "glm_logistic_iris"));
  print_endline "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg";

  let width_mean = (List.fold_left (+.) 0.0 petal_width) /. float_of_int (List.length petal_width) in
  let grid_n = 100 in
  let pl_min = (List.fold_left min infinity petal_length) -. 0.3 in
  let pl_max = (List.fold_left max neg_infinity petal_length) +. 0.3 in
  let grid_x = List.init grid_n (fun i -> pl_min +. (pl_max -. pl_min) *. float_of_int i /. float_of_int (grid_n - 1)) in
  let width_col = List.init grid_n (fun _ -> width_mean) in
  let grid = new_DataFrame C_void in
  ignore (invoke grid "add_numeric_column" (C_list [C_string "Petal.Length"; dv grid_x]));
  ignore (invoke grid "add_numeric_column" (C_list [C_string "Petal.Width"; dv width_col]));
  let curve_frame = invoke logit "predict_frame" (C_list [grid; C_string "confidence"]) in
  print_endline "\nPredicted-probability curve (first 5 rows):";
  print_endline (get_string (invoke curve_frame "to_string" (C_int64 5L)));

  print_endline "\n=================== Poisson regression (log link) ===================";
  let count = ref [] and sepal_width = ref [] and all_petal_length = ref [] in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    count := Float.round (get_float (invoke iris "numeric_at" (C_list [C_string "Sepal.Length"; i64]))) :: !count;
    sepal_width := get_float (invoke iris "numeric_at" (C_list [C_string "Sepal.Width"; i64])) :: !sepal_width;
    all_petal_length := get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; i64])) :: !all_petal_length
  done;
  let count_data = new_DataFrame C_void in
  ignore (invoke count_data "add_numeric_column" (C_list [C_string "Sepal.Width"; dv (List.rev !sepal_width)]));
  ignore (invoke count_data "add_numeric_column" (C_list [C_string "Petal.Length"; dv (List.rev !all_petal_length)]));
  ignore (invoke count_data "add_numeric_column" (C_list [C_string "count"; dv (List.rev !count)]));

  let poisson = new_GLM (C_list [count_data; C_string "count ~ Sepal.Width + Petal.Length"; C_string "poisson"]) in
  ignore (invoke poisson "print_summary" C_void)
