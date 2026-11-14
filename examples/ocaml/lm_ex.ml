(* CMD: just TARGET=lm_ex run-ocaml *)
(* Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions. *)
open Datamunge

let () =
  let cars = DataFrame.create () in
  DataFrame.add_numeric_column cars "hp"
    [| 110.0; 110.0; 93.0; 110.0; 175.0; 105.0; 245.0; 62.0; 95.0; 123.0 |];
  DataFrame.add_numeric_column cars "wt"
    [| 2.62; 2.875; 2.32; 3.215; 3.44; 3.46; 3.57; 3.19; 3.15; 3.44 |];
  DataFrame.add_string_column cars "transmission"
    [ "manual"; "manual"; "manual"; "automatic"; "automatic";
      "automatic"; "automatic"; "automatic"; "automatic"; "automatic" ];
  DataFrame.add_numeric_column cars "mpg"
    [| 21.0; 21.0; 22.8; 21.4; 18.7; 18.1; 14.3; 24.4; 22.8; 19.2 |];

  print_endline "Fitting: mpg ~ hp + wt + transmission\n";
  let model = LM.create cars "mpg ~ hp + wt + transmission" in
  LM.print_summary model;

  Printf.printf "\nSequential ANOVA:\n%s\n" (DataFrame.to_string (LM.anova model));

  let newcars = DataFrame.create () in
  DataFrame.add_numeric_column newcars "hp" [| 150.0; 90.0 |];
  DataFrame.add_numeric_column newcars "wt" [| 3.0; 2.5 |];
  DataFrame.add_string_column newcars "transmission" [ "manual"; "automatic" ];
  let predictions = LM.predict_frame ~interval_kind:"confidence" model newcars in
  Printf.printf "\nPredictions with 95%% confidence intervals:\n%s\n" (DataFrame.to_string predictions);

  LM.save_diagnostic_plots model "lm_ex_diagnostics";
  print_endline "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg"
