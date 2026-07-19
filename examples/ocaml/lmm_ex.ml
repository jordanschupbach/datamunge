open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let gauss () =
  let u1 = Random.float 1.0 and u2 = Random.float 1.0 in
  sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let () =
  print_endline "=================== Random intercept on a real dataset (penguins) ===================";
  let penguins = _DataFrame_penguins C_void in
  let species_model = new_LMM (C_list [penguins; C_string "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"]) in
  ignore (invoke species_model "print_summary" C_void);

  print_endline "\n=================== Random intercept + slope on a simulated multi-school dataset ===================";
  Random.init 2024;
  let n_schools = 30 in
  let school_intercept = Array.init n_schools (fun _ -> gauss () *. 6.0) in
  let school_slope = Array.init n_schools (fun _ -> gauss () *. 1.2) in

  let true_intercept = 60.0 and true_slope = 3.0 in
  let school = ref [] and study_hours = ref [] and score = ref [] in
  for s = 0 to n_schools - 1 do
    let n_students = 15 + Random.int 21 in
    for _ = 1 to n_students do
      let hours = Random.float 10.0 in
      let noise = gauss () *. 4.0 in
      let s_val = true_intercept +. school_intercept.(s) +. (true_slope +. school_slope.(s)) *. hours +. noise in
      school := float_of_int s :: !school;
      study_hours := hours :: !study_hours;
      score := s_val :: !score
    done
  done;

  let df = new_DataFrame C_void in
  ignore (invoke df "add_numeric_column" (C_list [C_string "school"; dv (List.rev !school)]));
  ignore (invoke df "add_numeric_column" (C_list [C_string "study_hours"; dv (List.rev !study_hours)]));
  ignore (invoke df "add_numeric_column" (C_list [C_string "score"; dv (List.rev !score)]));

  let model = new_LMM (C_list [df; C_string "score ~ study_hours + (1 + study_hours | school)"]) in
  ignore (invoke model "print_summary" C_void);

  Printf.printf "\nTrue generating values: intercept=%g, slope=%g, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n" true_intercept true_slope;

  print_endline "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---";
  let group_labels = invoke model "group_labels" C_void in
  List.iter (fun idx ->
    let re = invoke model "random_effects_for_group" (C_int64 (Int64.of_int idx)) in
    Printf.printf "school %s: intercept shift=%g, slope shift=%g\n"
      (get_string (invoke group_labels "[]" (C_int idx)))
      (get_float (invoke re "[]" (C_int 0))) (get_float (invoke re "[]" (C_int 1)))
  ) [0; 1; 2];

  print_endline "\n--- Prediction: population-level vs. school-adjusted ---";
  let newdata_population = new_DataFrame C_void in
  ignore (invoke newdata_population "add_numeric_column" (C_list [C_string "study_hours"; dv [5.0]]));
  let newdata_school0 = new_DataFrame C_void in
  ignore (invoke newdata_school0 "add_numeric_column" (C_list [C_string "study_hours"; dv [5.0]]));
  ignore (invoke newdata_school0 "add_numeric_column" (C_list [C_string "school"; dv [0.0]]));
  Printf.printf "5 study hours, unseen school:      %g (fixed effects only)\n" (get_float (invoke (invoke model "predict" newdata_population) "[]" (C_int 0)));
  Printf.printf "5 study hours, school 0 (known):    %g (fixed effects + school 0's BLUP)\n" (get_float (invoke (invoke model "predict" newdata_school0) "[]" (C_int 0)))
