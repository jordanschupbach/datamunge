(* CMD: just TARGET=tree_iris_ex run-ocaml *)
(* Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots. *)
open Datamunge

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let formula = "Species ~ Petal.Length + Petal.Width" in
  let model = DecisionTreeClassifier.create iris formula in
  DecisionTreeClassifier.print_summary model;

  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (DataFrame.to_string (DecisionTreeClassifier.confusion_matrix model));

  print_endline "\nMisclassified rows:";
  let predictions = DecisionTreeClassifier.predict model iris in
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
    (100.0 *. float_of_int !misclassified /. float_of_int n);

  RPlot.save
    (DecisionTreeClassifier.plot_classification model iris "Petal.Length" "Petal.Width")
    "tree_iris_classification.svg";
  RPlot.save
    (DecisionTreeClassifier.plot_decision_regions model "Petal.Length" "Petal.Width")
    "tree_iris_decision_regions.svg";
  print_endline "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg";

  (* A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision
     boundary. *)
  let shallow = DecisionTreeClassifier.create ~max_depth:2 iris formula in
  Printf.printf "\nDepth-2 tree training accuracy: %g%% (%d leaves)\n"
    (DecisionTreeClassifier.training_accuracy shallow *. 100.)
    (DecisionTreeClassifier.leaf_count shallow);
  RPlot.save
    (DecisionTreeClassifier.plot_decision_regions shallow "Petal.Length" "Petal.Width")
    "tree_iris_decision_regions_depth2.svg";
  print_endline "Saved tree_iris_decision_regions_depth2.svg"
