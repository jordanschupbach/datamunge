(* CMD: just TARGET=lmm_ex run-ocaml *)
(* Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools). *)
open Datamunge

(* One draw from datamunge's own generator (seeded below), so the simulated data (and the fit
   below) differ from languages that use their own generator. *)
let gauss () = (rnorm 0.0 1.0 1).(0)
let uniform () = (runif 0.0 1.0 1).(0)

let () =
  set_seed 2024;

  print_endline
    "=================== Random intercept on a real dataset (penguins) ===================";
  let penguin_model =
    LMM.create (Data.penguins ()) "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"
  in
  LMM.print_summary penguin_model;

  print_endline
    "\n=================== Random intercept + slope on a simulated multi-school dataset \
     ===================";
  (* 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
     and the return on an extra hour of study vary by school -- the textbook case for a
     random-intercept-and-slope model. *)
  let n_schools = 30 in
  let school_intercept = Array.init n_schools (fun _ -> gauss () *. 6.) in
  let school_slope = Array.init n_schools (fun _ -> gauss () *. 1.2) in
  let true_intercept = 60.0 and true_slope = 3.0 in

  let rows =
    List.init n_schools (fun s ->
        let n_students = 15 + int_of_float (uniform () *. 21.) in
        List.init n_students (fun _ ->
            let hours = uniform () *. 10. in
            let score =
              true_intercept +. school_intercept.(s)
              +. ((true_slope +. school_slope.(s)) *. hours)
              +. (gauss () *. 4.)
            in
            (float_of_int s, hours, score)))
    |> List.concat |> Array.of_list
  in
  let df = DataFrame.create () in
  DataFrame.add_numeric_column df "school" (Array.map (fun (s, _, _) -> s) rows);
  DataFrame.add_numeric_column df "study_hours" (Array.map (fun (_, h, _) -> h) rows);
  DataFrame.add_numeric_column df "score" (Array.map (fun (_, _, y) -> y) rows);

  let model = LMM.create df "score ~ study_hours + (1 + study_hours | school)" in
  LMM.print_summary model;

  Printf.printf
    "\nTrue generating values: intercept=%.1f, slope=%.1f, random-intercept SD=6.0, random-slope \
     SD=1.2, residual SD=4.0\n"
    true_intercept true_slope;

  print_endline "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---";
  let labels = Array.of_list (LMM.group_labels model) in
  for idx = 0 to 2 do
    let shifts = LMM.random_effects_for_group model idx in
    Printf.printf "school %s: intercept shift=%g, slope shift=%g\n" labels.(idx) shifts.(0)
      shifts.(1)
  done;

  print_endline "\n--- Prediction: population-level vs. school-adjusted ---";
  let population = DataFrame.create () and school0 = DataFrame.create () in
  DataFrame.add_numeric_column population "study_hours" [| 5.0 |];
  DataFrame.add_numeric_column school0 "study_hours" [| 5.0 |];
  DataFrame.add_numeric_column school0 "school" [| 0.0 |];
  Printf.printf "5 study hours, unseen school:      %g (fixed effects only)\n"
    (LMM.predict model population).(0);
  Printf.printf "5 study hours, school 0 (known):    %g (fixed effects + school 0's BLUP)\n"
    (LMM.predict model school0).(0)
