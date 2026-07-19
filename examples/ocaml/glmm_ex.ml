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

let gauss () =
  let u1 = Random.float 1.0 and u2 = Random.float 1.0 in
  sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let () =
  print_endline "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================";
  let penguins = _DataFrame_penguins C_void in
  let n = get_int (invoke penguins "nrows" C_void) in
  let is_male = ref [] and body_mass = ref [] and island = ref [] in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    let is_null c = get_bool (invoke penguins "is_null" (C_list [C_string c; i64])) in
    if not (is_null "sex" || is_null "body_mass_g" || is_null "island") then begin
      is_male := (if get_string (invoke penguins "string_at" (C_list [C_string "sex"; i64])) = "male" then 1.0 else 0.0) :: !is_male;
      body_mass := get_float (invoke penguins "numeric_at" (C_list [C_string "body_mass_g"; i64])) :: !body_mass;
      island := get_string (invoke penguins "string_at" (C_list [C_string "island"; i64])) :: !island
    end
  done;

  let sex_df = new_DataFrame C_void in
  ignore (invoke sex_df "add_numeric_column" (C_list [C_string "is_male"; dv (List.rev !is_male)]));
  ignore (invoke sex_df "add_numeric_column" (C_list [C_string "body_mass_g"; dv (List.rev !body_mass)]));
  ignore (invoke sex_df "add_string_column" (C_list [C_string "island"; sv (List.rev !island)]));

  let sex_model = new_GLMM (C_list [sex_df; C_string "is_male ~ body_mass_g + (1 | island)"; C_string "binomial"]) in
  ignore (invoke sex_model "print_summary" C_void);

  print_endline "\n=================== Poisson mixed model on simulated multi-site count data ===================";
  Random.init 4242;
  let n_stores = 25 in
  let store_effect = Array.init n_stores (fun _ -> gauss () *. 0.4) in

  let true_intercept = 2.0 and true_slope = 0.3 in
  let store = ref [] and promo = ref [] and visits = ref [] in
  for s = 0 to n_stores - 1 do
    let n_days = 15 + Random.int 11 in
    for _ = 1 to n_days do
      let promo_intensity = Random.float 3.0 in
      let lam = exp (true_intercept +. store_effect.(s) +. true_slope *. promo_intensity) in
      let l_thresh = exp (-. lam) in
      let k = ref 0 and p = ref 1.0 in
      let continue_loop = ref true in
      while !continue_loop do
        incr k;
        p := !p *. Random.float 1.0;
        if !p <= l_thresh then continue_loop := false
      done;
      store := float_of_int s :: !store;
      promo := promo_intensity :: !promo;
      visits := float_of_int (!k - 1) :: !visits
    done
  done;

  let df = new_DataFrame C_void in
  ignore (invoke df "add_numeric_column" (C_list [C_string "store"; dv (List.rev !store)]));
  ignore (invoke df "add_numeric_column" (C_list [C_string "promo"; dv (List.rev !promo)]));
  ignore (invoke df "add_numeric_column" (C_list [C_string "visits"; dv (List.rev !visits)]));

  let store_model = new_GLMM (C_list [df; C_string "visits ~ promo + (1 | store)"; C_string "poisson"]) in
  ignore (invoke store_model "print_summary" C_void);

  Printf.printf "\nTrue generating values: intercept=%g, slope=%g, random-intercept SD (log scale)=0.4\n" true_intercept true_slope;

  print_endline "\n--- BLUPs for a few stores ---";
  let group_labels = invoke store_model "group_labels" C_void in
  List.iter (fun idx ->
    let re = invoke store_model "random_effects_for_group" (C_int64 (Int64.of_int idx)) in
    Printf.printf "store %s: intercept shift=%g\n" (get_string (invoke group_labels "[]" (C_int idx))) (get_float (invoke re "[]" (C_int 0)))
  ) [0; 1; 2];

  print_endline "\n--- Prediction: population-level vs. store-adjusted ---";
  let newdata_population = new_DataFrame C_void in
  ignore (invoke newdata_population "add_numeric_column" (C_list [C_string "promo"; dv [1.5]]));
  let newdata_store0 = new_DataFrame C_void in
  ignore (invoke newdata_store0 "add_numeric_column" (C_list [C_string "promo"; dv [1.5]]));
  ignore (invoke newdata_store0 "add_numeric_column" (C_list [C_string "store"; dv [0.0]]));
  Printf.printf "promo=1.5, unseen store:   %g expected visits (fixed effects only)\n" (get_float (invoke (invoke store_model "predict" newdata_population) "[]" (C_int 0)));
  Printf.printf "promo=1.5, store 0 (known): %g expected visits (fixed effects + store 0's BLUP)\n" (get_float (invoke (invoke store_model "predict" newdata_store0) "[]" (C_int 0)))
