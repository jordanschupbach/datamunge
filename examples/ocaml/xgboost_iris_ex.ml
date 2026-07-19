open Swig
open Datamunge

let last_of v =
  let n = get_int (invoke v "size" C_void) in
  get_float (invoke v "[]" (C_int (n - 1)))

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n\n"
    (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));

  let model = new_XGBoostClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"]) in
  ignore (invoke model "print_summary" C_void);

  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (get_string (invoke (invoke model "confusion_matrix" C_void) "to_string" C_void));

  print_endline "\nMisclassified rows:";
  let predictions = invoke model "predict" iris in
  let n = get_int (invoke iris "nrows" C_void) in
  let misclassified = ref 0 in
  for i = 0 to n - 1 do
    let actual = get_string (invoke iris "string_at" (C_list [C_string "Species"; C_int64 (Int64.of_int i)])) in
    let pred = get_string (invoke predictions "[]" (C_int i)) in
    if pred <> actual then begin
      incr misclassified;
      Printf.printf "  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n" i
        (get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; C_int64 (Int64.of_int i)])))
        (get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Width"; C_int64 (Int64.of_int i)])))
        actual pred
    end
  done;
  Printf.printf "%d of %d misclassified (%g%%)\n" !misclassified n (100.0 *. float_of_int !misclassified /. float_of_int n);

  ignore (invoke (invoke model "plot_training_deviance" C_void) "save" (C_string "xgboost_iris_training_deviance.svg"));
  ignore (invoke (invoke model "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "xgboost_iris_decision_regions.svg"));
  print_endline "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg";

  let heavy = new_XGBoostClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"; C_int64 100L; C_double 0.3; C_int64 6L; C_double 50.0]) in
  Printf.printf "\nlambda=1 (default):   training accuracy=%g%%  deviance=%g\n"
    (get_float (invoke model "training_accuracy" C_void) *. 100.0) (last_of (invoke model "training_deviance" C_void));
  Printf.printf "lambda=50 (heavy L2): training accuracy=%g%%  deviance=%g\n"
    (get_float (invoke heavy "training_accuracy" C_void) *. 100.0) (last_of (invoke heavy "training_deviance" C_void));
  ignore (invoke (invoke heavy "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "xgboost_iris_decision_regions_heavy_lambda.svg"));
  print_endline "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
