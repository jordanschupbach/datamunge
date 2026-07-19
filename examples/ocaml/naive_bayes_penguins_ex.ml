open Swig
open Datamunge

let () =
  let penguins = _DataFrame_penguins C_void in
  Printf.printf "penguins: %d rows x %d cols\n\n"
    (get_int (invoke penguins "nrows" C_void)) (get_int (invoke penguins "ncols" C_void));

  let model = new_NaiveBayesClassifier (C_list [penguins; C_string "species ~ bill_length_mm + bill_depth_mm + island + sex"]) in
  ignore (invoke model "print_summary" C_void);

  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (get_string (invoke (invoke model "confusion_matrix" C_void) "to_string" C_void));

  print_endline "\nMisclassified rows:";
  let predictions = invoke model "predict" penguins in
  let n = get_int (invoke penguins "nrows" C_void) in
  let misclassified = ref 0 in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    let is_null c = get_bool (invoke penguins "is_null" (C_list [C_string c; i64])) in
    if not (is_null "species" || is_null "bill_length_mm" || is_null "bill_depth_mm" || is_null "island" || is_null "sex") then begin
      let actual = get_string (invoke penguins "string_at" (C_list [C_string "species"; i64])) in
      let pred = get_string (invoke predictions "[]" (C_int i)) in
      if pred <> actual then begin
        incr misclassified;
        Printf.printf "  row %d: bill_length=%g bill_depth=%g island=%s sex=%s  actual=%s  predicted=%s\n" i
          (get_float (invoke penguins "numeric_at" (C_list [C_string "bill_length_mm"; i64])))
          (get_float (invoke penguins "numeric_at" (C_list [C_string "bill_depth_mm"; i64])))
          (get_string (invoke penguins "string_at" (C_list [C_string "island"; i64])))
          (get_string (invoke penguins "string_at" (C_list [C_string "sex"; i64])))
          actual pred
      end
    end
  done;
  Printf.printf "%d misclassified (of %d rows, some incomplete)\n" !misclassified n;

  let bill_only = new_NaiveBayesClassifier (C_list [penguins; C_string "species ~ bill_length_mm + bill_depth_mm"]) in
  Printf.printf "\nbill-measurements-only model training accuracy: %g%%\n" (get_float (invoke bill_only "training_accuracy" C_void) *. 100.0);
  ignore (invoke (invoke bill_only "plot_classification" (C_list [penguins; C_string "bill_length_mm"; C_string "bill_depth_mm"])) "save" (C_string "naive_bayes_penguins_classification.svg"));
  ignore (invoke (invoke bill_only "plot_decision_regions" (C_list [C_string "bill_length_mm"; C_string "bill_depth_mm"])) "save" (C_string "naive_bayes_penguins_decision_regions.svg"));
  print_endline "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
