(* CMD: just TARGET=naive_bayes_penguins_ex run-ocaml *)
(* Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots. *)
open Datamunge

let () =
  (* Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
     when fitting, matching every other formula-based model in this library. *)
  let penguins = Data.penguins () in
  Printf.printf "penguins: %d rows x %d cols\n\n" (DataFrame.nrows penguins)
    (DataFrame.ncols penguins);

  (* A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
     predictors in one formula -- each modeled independently given the class, per the naive
     Bayes assumption. *)
  let model =
    NaiveBayesClassifier.create penguins "species ~ bill_length_mm + bill_depth_mm + island + sex"
  in
  NaiveBayesClassifier.print_summary model;

  Printf.printf "\nConfusion matrix (rows = actual, cols = predicted):\n%s\n"
    (DataFrame.to_string (NaiveBayesClassifier.confusion_matrix model));

  print_endline "\nMisclassified rows:";
  let predictions = Array.of_list (NaiveBayesClassifier.predict model penguins) in
  let columns = [ "species"; "bill_length_mm"; "bill_depth_mm"; "island"; "sex" ] in
  let misclassified = ref 0 in
  for i = 0 to DataFrame.nrows penguins - 1 do
    let complete = not (List.exists (fun column -> DataFrame.is_null penguins column i) columns) in
    let actual = DataFrame.string_at penguins "species" i in
    let predicted = predictions.(i) in
    if complete && predicted <> actual then begin
      incr misclassified;
      Printf.printf
        "  row %d: bill_length=%g bill_depth=%g island=%s sex=%s  actual=%s  predicted=%s\n" i
        (DataFrame.numeric_at penguins "bill_length_mm" i)
        (DataFrame.numeric_at penguins "bill_depth_mm" i)
        (DataFrame.string_at penguins "island" i)
        (DataFrame.string_at penguins "sex" i)
        actual predicted
    end
  done;
  Printf.printf "%d misclassified (of %d rows, some incomplete)\n" !misclassified
    (DataFrame.nrows penguins);

  (* plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
     two-predictor model (bill measurements alone) just for visualization. *)
  let bill_only =
    NaiveBayesClassifier.create penguins "species ~ bill_length_mm + bill_depth_mm"
  in
  Printf.printf "\nbill-measurements-only model training accuracy: %g%%\n"
    (100. *. NaiveBayesClassifier.training_accuracy bill_only);
  RPlot.save
    (NaiveBayesClassifier.plot_classification bill_only penguins "bill_length_mm" "bill_depth_mm")
    "naive_bayes_penguins_classification.svg";
  RPlot.save
    (NaiveBayesClassifier.plot_decision_regions bill_only "bill_length_mm" "bill_depth_mm")
    "naive_bayes_penguins_decision_regions.svg";
  print_endline
    "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"
