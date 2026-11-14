(* CMD: just TARGET=forest_iris_ex run-ocaml *)
(* Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy. *)

open Datamunge

let formula = "Species ~ Petal.Length + Petal.Width"

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let model = RandomForestClassifier.create iris formula in
  RandomForestClassifier.print_summary model;

  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (DataFrame.to_string (RandomForestClassifier.confusion_matrix model));

  print_endline "\nMisclassified rows:";
  let predictions = RandomForestClassifier.predict model iris in
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

  RPlot.save
    (RandomForestClassifier.plot_classification model iris "Petal.Length" "Petal.Width")
    "forest_iris_classification.svg";
  RPlot.save
    (RandomForestClassifier.plot_decision_regions model "Petal.Length" "Petal.Width")
    "forest_iris_decision_regions.svg";
  print_endline "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg";

  (* A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees are
     added -- the defining random forest effect a single decision tree cannot demonstrate. *)
  let small_forest = RandomForestClassifier.create ~n_trees:5 iris formula in
  Printf.printf "\n5-tree forest:   training accuracy=%g%%  OOB accuracy=%g%%\n"
    (RandomForestClassifier.training_accuracy small_forest *. 100.0)
    (RandomForestClassifier.oob_accuracy small_forest *. 100.0);
  Printf.printf "100-tree forest: training accuracy=%g%%  OOB accuracy=%g%%\n"
    (RandomForestClassifier.training_accuracy model *. 100.0)
    (RandomForestClassifier.oob_accuracy model *. 100.0);
  RPlot.save
    (RandomForestClassifier.plot_decision_regions small_forest "Petal.Length" "Petal.Width")
    "forest_iris_decision_regions_5trees.svg";
  print_endline "Saved forest_iris_decision_regions_5trees.svg"
