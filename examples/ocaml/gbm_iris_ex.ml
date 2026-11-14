(* CMD: just TARGET=gbm_iris_ex run-ocaml *)
(* Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds. *)

open Datamunge

let formula = "Species ~ Petal.Length + Petal.Width"

let final_deviance model =
  let deviance = GBMClassifier.training_deviance model in
  deviance.(Array.length deviance - 1)

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let model = GBMClassifier.create iris formula in
  GBMClassifier.print_summary model;

  Printf.printf "\nConfusion matrix (rows = actual, cols = predicted):\n%s\n"
    (DataFrame.to_string (GBMClassifier.confusion_matrix model));

  print_endline "\nMisclassified rows:";
  let predictions = GBMClassifier.predict model iris in
  let n = DataFrame.nrows iris in
  let misclassified = ref 0 in
  List.iteri
    (fun i predicted ->
      let actual = DataFrame.string_at iris "Species" i in
      if predicted <> actual then begin
        incr misclassified;
        Printf.printf "  row %d: Petal.Length=%g Petal.Width=%g  actual=%s  predicted=%s\n" i
          (DataFrame.numeric_at iris "Petal.Length" i)
          (DataFrame.numeric_at iris "Petal.Width" i)
          actual predicted
      end)
    predictions;
  Printf.printf "%d of %d misclassified (%g%%)\n" !misclassified n
    (100.0 *. float !misclassified /. float n);

  RPlot.save (GBMClassifier.plot_training_deviance model) "gbm_iris_training_deviance.svg";
  RPlot.save
    (GBMClassifier.plot_decision_regions model "Petal.Length" "Petal.Width")
    "gbm_iris_decision_regions.svg";
  print_endline "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg";

  (* A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away at
     the training loss, gradually sharpening the decision boundary. *)
  let few = GBMClassifier.create ~n_trees:5 iris formula in
  Printf.printf "\n5-round ensemble:   training accuracy=%g%%  deviance=%g\n"
    (100.0 *. GBMClassifier.training_accuracy few)
    (final_deviance few);
  Printf.printf "100-round ensemble: training accuracy=%g%%  deviance=%g\n"
    (100.0 *. GBMClassifier.training_accuracy model)
    (final_deviance model);
  RPlot.save
    (GBMClassifier.plot_decision_regions few "Petal.Length" "Petal.Width")
    "gbm_iris_decision_regions_5rounds.svg";
  print_endline "Saved gbm_iris_decision_regions_5rounds.svg"
