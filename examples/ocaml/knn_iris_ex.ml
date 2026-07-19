open Swig
open Datamunge

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n\n"
    (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));

  let model = new_KNNClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"]) in
  ignore (invoke model "print_summary" C_void);

  print_endline "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):";
  print_endline (get_string (invoke (invoke model "confusion_matrix" C_void) "to_string" C_void));

  print_endline "\nLeave-one-out misclassified rows:";
  let fitted = invoke model "predict" iris in
  let n = get_int (invoke iris "nrows" C_void) in
  let misclassified = ref 0 in
  for i = 0 to n - 1 do
    let actual = get_string (invoke iris "string_at" (C_list [C_string "Species"; C_int64 (Int64.of_int i)])) in
    let pred = get_string (invoke fitted "[]" (C_int i)) in
    if pred <> actual then begin
      incr misclassified;
      Printf.printf "  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n" i
        (get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; C_int64 (Int64.of_int i)])))
        (get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Width"; C_int64 (Int64.of_int i)])))
        actual pred
    end
  done;
  Printf.printf "%d of %d misclassified (%g%%)\n" !misclassified n (100.0 *. float_of_int !misclassified /. float_of_int n);

  ignore (invoke (invoke model "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "knn_iris_decision_regions_k5.svg"));
  print_endline "\nSaved knn_iris_decision_regions_k5.svg";

  let k1 = new_KNNClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"; C_int64 1L]) in
  Printf.printf "\nk=1  leave-one-out accuracy: %g%%\n" (get_float (invoke k1 "training_accuracy" C_void) *. 100.0);
  ignore (invoke (invoke k1 "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "knn_iris_decision_regions_k1.svg"));
  print_endline "Saved knn_iris_decision_regions_k1.svg";

  let k25 = new_KNNClassifier (C_list [iris; C_string "Species ~ Petal.Length + Petal.Width"; C_int64 25L]) in
  Printf.printf "\nk=25 leave-one-out accuracy: %g%%\n" (get_float (invoke k25 "training_accuracy" C_void) *. 100.0);
  ignore (invoke (invoke k25 "plot_decision_regions" (C_list [C_string "Petal.Length"; C_string "Petal.Width"])) "save" (C_string "knn_iris_decision_regions_k25.svg"));
  print_endline "Saved knn_iris_decision_regions_k25.svg"
