open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n\n"
    (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));

  print_endline "=== RBF kernel (default) ===";
  let rbf_model = new_SVM (C_list [iris; C_string "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"]) in
  ignore (invoke rbf_model "print_summary" C_void);
  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (get_string (invoke (invoke rbf_model "confusion_matrix" C_void) "to_string" C_void));

  print_endline "\n=== Linear kernel, for comparison ===";
  let linear_model = new_SVM (C_list [iris; C_string "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"; C_string "linear"]) in
  Printf.printf "Training accuracy: %g%%\n" (get_float (invoke linear_model "training_accuracy" C_void) *. 100.0);
  Printf.printf "Support vectors: %d\n" (get_int (invoke linear_model "num_support_vectors" C_void));

  let newdata = new_DataFrame C_void in
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Sepal.Length"; dv [5.1; 6.0; 6.5; 6.2]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Sepal.Width"; dv [3.5; 2.7; 3.0; 2.8]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Petal.Length"; dv [1.4; 4.5; 5.5; 4.8]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Petal.Width"; dv [0.2; 1.5; 2.0; 1.8]]));

  print_endline "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):";
  print_endline (get_string (invoke (invoke rbf_model "predict_frame" newdata) "to_string" C_void))
