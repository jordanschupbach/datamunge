open Swig
open Datamunge

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n\n"
    (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));

  let model = new_DecisionTreeClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"]) in
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

  ignore (invoke (invoke model "plot_classification" (C_list [iris; C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "tree_iris_classification.svg"));
  ignore (invoke (invoke model "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "tree_iris_decision_regions.svg"));
  print_endline "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg";

  let shallow = new_DecisionTreeClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"; C_int64 2L]) in
  Printf.printf "\nDepth-2 tree training accuracy: %g%% (%d leaves)\n"
    (get_float (invoke shallow "training_accuracy" C_void) *. 100.0) (get_int (invoke shallow "leaf_count" C_void));
  ignore (invoke (invoke shallow "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "tree_iris_decision_regions_depth2.svg"));
  print_endline "Saved tree_iris_decision_regions_depth2.svg"
