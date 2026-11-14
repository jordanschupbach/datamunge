(* CMD: just TARGET=svm_iris_ex run-ocaml *)
(* Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions. *)
open Datamunge

let () =
  let formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width" in
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  print_endline "=== RBF kernel (default) ===";
  let rbf_model = SVM.create iris formula in
  SVM.print_summary rbf_model;
  Printf.printf "\nConfusion matrix (rows = actual, cols = predicted):\n%s\n"
    (DataFrame.to_string (SVM.confusion_matrix rbf_model));

  print_endline "\n=== Linear kernel, for comparison ===";
  let linear_model = SVM.create ~kernel:"linear" iris formula in
  Printf.printf "Training accuracy: %g%%\n" (SVM.training_accuracy linear_model *. 100.);
  Printf.printf "Support vectors: %d\n" (SVM.num_support_vectors linear_model);

  let newdata = DataFrame.create () in
  DataFrame.add_numeric_column newdata "Sepal.Length" [| 5.1; 6.0; 6.5; 6.2 |];
  DataFrame.add_numeric_column newdata "Sepal.Width" [| 3.5; 2.7; 3.0; 2.8 |];
  DataFrame.add_numeric_column newdata "Petal.Length" [| 1.4; 4.5; 5.5; 4.8 |];
  DataFrame.add_numeric_column newdata "Petal.Width" [| 0.2; 1.5; 2.0; 1.8 |];
  Printf.printf "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n%s\n"
    (DataFrame.to_string (SVM.predict_frame rbf_model newdata))
