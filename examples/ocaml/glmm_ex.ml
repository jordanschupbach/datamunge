(* CMD: just TARGET=glmm_ex run-ocaml *)
(* Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data. *)
open Datamunge

(* OCaml's own generator, so the simulated data (and the fit below) differ from other languages'. *)
let () = Random.init 4242

(* Box-Muller; 1 - u keeps log() away from 0. *)
let gauss () =
  let u = 1.0 -. Random.float 1.0 in
  sqrt (-2.0 *. log u) *. cos (2.0 *. Float.pi *. Random.float 1.0)

(* Knuth's Poisson sampler. *)
let rpois lambda =
  let threshold = exp (-.lambda) in
  let rec loop k p =
    let p = p *. Random.float 1.0 in
    if p <= threshold then k else loop (k + 1) p
  in
  loop 0 1.0

let () =
  print_endline
    "=================== Binomial (logistic) mixed model on a real dataset (penguins) \
     ===================";
  (* Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
     confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
     is not tied to location, so this is a well-behaved fit. *)
  let penguins = Data.penguins () in
  let rows =
    List.init (DataFrame.nrows penguins) Fun.id
    |> List.filter (fun i ->
           [ "sex"; "body_mass_g"; "island" ]
           |> List.for_all (fun c -> not (DataFrame.is_null penguins c i)))
  in
  let sex_df = DataFrame.create () in
  DataFrame.add_numeric_column sex_df "is_male"
    (Array.of_list
       (List.map
          (fun i -> if DataFrame.string_at penguins "sex" i = "male" then 1.0 else 0.0)
          rows));
  DataFrame.add_numeric_column sex_df "body_mass_g"
    (Array.of_list (List.map (DataFrame.numeric_at penguins "body_mass_g") rows));
  DataFrame.add_string_column sex_df "island"
    (List.map (DataFrame.string_at penguins "island") rows);
  let sex_model = GLMM.create ~family:"binomial" sex_df "is_male ~ body_mass_g + (1 | island)" in
  GLMM.print_summary sex_model;

  print_endline
    "\n\
     =================== Poisson mixed model on simulated multi-site count data \
     ===================";
  (* 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
     the baseline traffic and the promo's effectiveness vary by store. *)
  let n_stores = 25 in
  let store_effect = Array.init n_stores (fun _ -> gauss () *. 0.4) in
  let true_intercept = 2.0 and true_slope = 0.3 in

  let store = ref [] and promo = ref [] and visits = ref [] in
  for s = 0 to n_stores - 1 do
    let n_days = 15 + Random.int 11 in
    for _ = 1 to n_days do
      let promo_intensity = Random.float 3.0 in
      let lambda = exp (true_intercept +. store_effect.(s) +. (true_slope *. promo_intensity)) in
      store := float_of_int s :: !store;
      promo := promo_intensity :: !promo;
      visits := float_of_int (rpois lambda) :: !visits
    done
  done;
  let column values = Array.of_list (List.rev values) in
  let df = DataFrame.create () in
  DataFrame.add_numeric_column df "store" (column !store);
  DataFrame.add_numeric_column df "promo" (column !promo);
  DataFrame.add_numeric_column df "visits" (column !visits);
  let store_model = GLMM.create ~family:"poisson" df "visits ~ promo + (1 | store)" in

  GLMM.print_summary store_model;

  Printf.printf
    "\nTrue generating values: intercept=%g, slope=%g, random-intercept SD (log scale)=0.4\n"
    true_intercept true_slope;

  print_endline "\n--- BLUPs for a few stores ---";
  List.iteri
    (fun idx label ->
      if idx < 3 then
        Printf.printf "store %s: intercept shift=%g\n" label
          (GLMM.random_effects_for_group store_model idx).(0))
    (GLMM.group_labels store_model);

  print_endline "\n--- Prediction: population-level vs. store-adjusted ---";
  let population = DataFrame.create () in
  let store0 = DataFrame.create () in
  DataFrame.add_numeric_column population "promo" [| 1.5 |];
  DataFrame.add_numeric_column store0 "promo" [| 1.5 |];
  DataFrame.add_numeric_column store0 "store" [| 0.0 |];
  Printf.printf "promo=1.5, unseen store:   %g expected visits (fixed effects only)\n"
    (GLMM.predict store_model population).(0);
  Printf.printf
    "promo=1.5, store 0 (known): %g expected visits (fixed effects + store 0's BLUP)\n"
    (GLMM.predict store_model store0).(0)
