(* CMD: just TARGET=lda_iris_ex run-ocaml *)
(* Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot. *)
open Datamunge

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let model =
    LDA.create iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
  in
  LDA.print_summary model;

  Printf.printf "\nConfusion matrix (rows = actual, cols = predicted):\n%s\n"
    (DataFrame.to_string (LDA.confusion_matrix model));

  let newdata = DataFrame.create () in
  DataFrame.add_numeric_column newdata "Sepal.Length" [| 5.1; 6.0; 6.5; 6.2 |];
  DataFrame.add_numeric_column newdata "Sepal.Width" [| 3.5; 2.7; 3.0; 2.8 |];
  DataFrame.add_numeric_column newdata "Petal.Length" [| 1.4; 4.5; 5.5; 4.8 |];
  DataFrame.add_numeric_column newdata "Petal.Width" [| 0.2; 1.5; 2.0; 1.8 |];
  Printf.printf "\nPredictions for new flowers:\n%s\n"
    (DataFrame.to_string (LDA.predict_frame model newdata));

  LDA.save_discriminant_plot model "lda_iris_discriminants.svg";
  print_endline "\nSaved discriminant plot as lda_iris_discriminants.svg"
