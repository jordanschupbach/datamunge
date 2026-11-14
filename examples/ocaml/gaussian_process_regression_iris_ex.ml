(* CMD: just TARGET=gaussian_process_regression_iris_ex run-ocaml *)
(* Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals. *)

open Datamunge

let formula = "Petal.Length ~ Petal.Width"

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n" (DataFrame.nrows iris) (DataFrame.ncols iris);
  Printf.printf "formula: %s\n\n" formula;

  (* Both the length scale and the noise ratio are auto-selected by maximizing the exact log
     marginal likelihood. *)
  let model = GaussianProcessRegression.create iris formula in
  GaussianProcessRegression.print_summary model;

  RPlot.save (GaussianProcessRegression.plot_fit model iris) "gpr_iris_fit.svg";
  RPlot.save
    (GaussianProcessRegression.plot_length_scale_profile model)
    "gpr_iris_length_scale_profile.svg";
  print_endline "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg";

  (* Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
     interval at every point -- including far outside the training data, where it should widen
     substantially as the model's uncertainty grows. *)
  let query = DataFrame.create () in
  DataFrame.add_numeric_column query "Petal.Width" [| 0.2; 1.3; 2.5; 10.0 |];
  let detail = GaussianProcessRegression.predict_frame ~interval_kind:"confidence" model query in
  print_endline "\nPredictions with 95% confidence intervals:";
  print_endline (DataFrame.to_string detail);
  print_endline
    "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n\
    \ interval is than the in-range predictions.)"
