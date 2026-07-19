open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let sv values =
  let v = new_SVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_string x))) values;
  v

let () =
  let hp = [110.0; 110.0; 93.0; 110.0; 175.0; 105.0; 245.0; 62.0; 95.0; 123.0] in
  let wt = [2.62; 2.875; 2.32; 3.215; 3.44; 3.46; 3.57; 3.19; 3.15; 3.44] in
  let transmission = ["manual"; "manual"; "manual"; "automatic"; "automatic";
                       "automatic"; "automatic"; "automatic"; "automatic"; "automatic"] in
  let mpg = [21.0; 21.0; 22.8; 21.4; 18.7; 18.1; 14.3; 24.4; 22.8; 19.2] in

  let cars = new_DataFrame C_void in
  ignore (invoke cars "add_numeric_column" (C_list [C_string "hp"; dv hp]));
  ignore (invoke cars "add_numeric_column" (C_list [C_string "wt"; dv wt]));
  ignore (invoke cars "add_string_column" (C_list [C_string "transmission"; sv transmission]));
  ignore (invoke cars "add_numeric_column" (C_list [C_string "mpg"; dv mpg]));

  print_endline "Fitting: mpg ~ hp + wt + transmission\n";
  let model = new_LM (C_list [cars; C_string "mpg ~ hp + wt + transmission"]) in
  ignore (invoke model "print_summary" C_void);

  print_endline "\nSequential ANOVA:";
  print_endline (get_string (invoke (invoke model "anova" C_void) "to_string" C_void));

  let newcars = new_DataFrame C_void in
  ignore (invoke newcars "add_numeric_column" (C_list [C_string "hp"; dv [150.0; 90.0]]));
  ignore (invoke newcars "add_numeric_column" (C_list [C_string "wt"; dv [3.0; 2.5]]));
  ignore (invoke newcars "add_string_column" (C_list [C_string "transmission"; sv ["manual"; "automatic"]]));

  let frame = invoke model "predict_frame" (C_list [newcars; C_string "confidence"]) in
  print_endline "\nPredictions with 95% confidence intervals:";
  print_endline (get_string (invoke frame "to_string" C_void));

  ignore (invoke model "save_diagnostic_plots" (C_string "lm_ex_diagnostics"));
  print_endline "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg"
