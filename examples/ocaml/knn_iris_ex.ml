(* CMD: just TARGET=knn_iris_ex run-ocaml *)
(* k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k. *)
open Datamunge

let formula = "Species ~ Petal.Length + Petal.Width"

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n\n" (DataFrame.nrows iris) (DataFrame.ncols iris);

  let model = KNNClassifier.create iris formula in
  KNNClassifier.print_summary model;

  print_endline "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):";
  print_endline (DataFrame.to_string (KNNClassifier.confusion_matrix model));

  print_endline "\nLeave-one-out misclassified rows:";
  let fitted = KNNClassifier.predict model iris in
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
    fitted;
  Printf.printf "%d of %d misclassified (%g%%)\n" !misclassified n
    (100.0 *. float_of_int !misclassified /. float_of_int n);

  Plot.save
    (KNNClassifier.plot_decision_regions model "Petal.Length" "Petal.Width")
    "knn_iris_decision_regions_k5.svg";
  print_endline "\nSaved knn_iris_decision_regions_k5.svg";

  (* k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
     every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
     underfit boundary). *)
  List.iter
    (fun k ->
      let knn = KNNClassifier.create ~k iris formula in
      let file = Printf.sprintf "knn_iris_decision_regions_k%d.svg" k in
      Printf.printf "\nk=%-2d leave-one-out accuracy: %g%%\n" k
        (KNNClassifier.training_accuracy knn *. 100.0);
      Plot.save (KNNClassifier.plot_decision_regions knn "Petal.Length" "Petal.Width") file;
      Printf.printf "Saved %s\n" file)
    [ 1; 25 ]
