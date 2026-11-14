(* CMD: just TARGET=xgboost_iris_ex run-ocaml *)
(* XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison. *)
open Datamunge

let last values = values.(Array.length values - 1)

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let formula = "Species ~ Petal.Length + Petal.Width" in
  let model = XGBoostClassifier.create iris formula in
  XGBoostClassifier.print_summary model;

  Printf.printf "\nConfusion matrix (rows = actual, cols = predicted):\n%s\n"
    (DataFrame.to_string (XGBoostClassifier.confusion_matrix model));

  print_endline "\nMisclassified rows:";
  let predictions = Array.of_list (XGBoostClassifier.predict model iris) in
  let n = DataFrame.nrows iris in
  let misclassified = ref 0 in
  for i = 0 to n - 1 do
    let actual = DataFrame.string_at iris "Species" i in
    let predicted = predictions.(i) in
    if predicted <> actual then begin
      incr misclassified;
      Printf.printf "  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n" i
        (DataFrame.numeric_at iris "Petal.Length" i)
        (DataFrame.numeric_at iris "Petal.Width" i)
        actual predicted
    end
  done;
  Printf.printf "%d of %d misclassified (%g%%)\n" !misclassified n
    (100.0 *. float_of_int !misclassified /. float_of_int n);

  RPlot.save (XGBoostClassifier.plot_training_deviance model) "xgboost_iris_training_deviance.svg";
  RPlot.save
    (XGBoostClassifier.plot_decision_regions model "Petal.Length" "Petal.Width")
    "xgboost_iris_decision_regions.svg";
  print_endline "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg";

  (* XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
     grows the same deep trees but keeps every leaf weight small, producing a much smoother
     decision boundary than the lightly-regularized default. *)
  let heavy =
    XGBoostClassifier.create ~n_trees:100 ~learning_rate:0.3 ~max_depth:6 ~lambda:50.0 iris formula
  in
  Printf.printf "\nlambda=1 (default):   training accuracy=%g%%  deviance=%g\n"
    (100. *. XGBoostClassifier.training_accuracy model)
    (last (XGBoostClassifier.training_deviance model));
  Printf.printf "lambda=50 (heavy L2): training accuracy=%g%%  deviance=%g\n"
    (100. *. XGBoostClassifier.training_accuracy heavy)
    (last (XGBoostClassifier.training_deviance heavy));
  RPlot.save
    (XGBoostClassifier.plot_decision_regions heavy "Petal.Length" "Petal.Width")
    "xgboost_iris_decision_regions_heavy_lambda.svg";
  print_endline "Saved xgboost_iris_decision_regions_heavy_lambda.svg"
