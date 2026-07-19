open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let formula = "Petal.Length ~ Petal.Width"

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n" (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));
  Printf.printf "formula: %s\n\n" formula;

  let model = new_GaussianProcessRegression (C_list [iris; C_string formula]) in
  ignore (invoke model "print_summary" C_void);

  ignore (invoke (invoke model "plot_fit" iris) "save" (C_string "gpr_iris_fit.svg"));
  ignore (invoke (invoke model "plot_length_scale_profile" C_void) "save" (C_string "gpr_iris_length_scale_profile.svg"));
  print_endline "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg";

  let query = new_DataFrame C_void in
  ignore (invoke query "add_numeric_column" (C_list [C_string "Petal.Width"; dv [0.2; 1.3; 2.5; 10.0]]));
  let detail = invoke model "predict_frame" (C_list [query; C_string "confidence"]) in
  print_endline "\nPredictions with 95% confidence intervals:";
  print_endline (get_string (invoke detail "to_string" C_void));
  print_endline "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)"
