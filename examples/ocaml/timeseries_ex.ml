open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

(* ARIMA/ExponentialSmoothing const-ref vector<double> returns come back as an unrecognized
   raw pointer -- re-wrap through the public create_std_..._from_ptr helper before using. *)
let fixdv raw = create_std_xxvector_xx_ldbrace_xx_lparendouble_xx_rparen_xx_rdbrace_from_ptr raw

let dv_to_list v =
  let n = get_int (invoke v "size" C_void) in
  List.init n (fun i -> get_float (invoke v "[]" (C_int i)))

let str_of_dv v = String.concat ", " (List.map (Printf.sprintf "%g") (dv_to_list v))

let gauss () =
  let u1 = Random.float 1.0 and u2 = Random.float 1.0 in
  sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

let () =
  print_endline "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================";
  Random.init 42;
  let y = ref [] in
  let level = ref 100.0 and prev_shock = ref 0.0 in
  for _ = 1 to 150 do
    let shock = gauss () in
    level := !level +. 0.3 +. 0.4 *. !prev_shock +. shock;
    y := !level :: !y;
    prev_shock := shock
  done;

  let options = new_ARIMAOptions C_void in
  ignore (invoke options "[p]" (C_int64 1L));
  ignore (invoke options "[d]" (C_int64 1L));
  ignore (invoke options "[q]" (C_int64 1L));
  ignore (invoke options "[de_population_size]" (C_int64 80L));
  ignore (invoke options "[de_max_generations]" (C_int64 400L));
  let model = new_ARIMA (C_list [dv (List.rev !y); options]) in

  let ar = fixdv (invoke model "ar_coefficients" C_void) in
  let ma = fixdv (invoke model "ma_coefficients" C_void) in
  Printf.printf "AR coefficient: %g\n" (get_float (invoke ar "[]" (C_int 0)));
  Printf.printf "MA coefficient: %g\n" (get_float (invoke ma "[]" (C_int 0)));
  Printf.printf "sigma^2: %g, AIC: %g, BIC: %g\n" (get_float (invoke model "sigma2" C_void)) (get_float (invoke model "aic" C_void)) (get_float (invoke model "bic" C_void));

  let pair = invoke model "forecast_with_intervals" (C_int64 6L) in
  let point = fixdv (invoke pair "[first]" C_void) in
  let se = fixdv (invoke pair "[second]" C_void) in
  Printf.printf "6-step forecast: %s\n" (str_of_dv point);
  Printf.printf "forecast std. errors: %s\n" (str_of_dv se);

  print_endline "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================";
  Random.init 7;
  let s = ref [] in
  let prev = ref 0.0 in
  for i = 0 to 119 do
    prev := 0.5 *. !prev +. gauss ();
    s := (20.0 +. 0.2 *. float_of_int i +. 5.0 *. sin (2.0 *. Float.pi *. float_of_int i /. 12.0) +. !prev) :: !s
  done;

  let options2 = new_ARIMAOptions C_void in
  ignore (invoke options2 "[p]" (C_int64 1L));
  ignore (invoke options2 "[seasonal_p]" (C_int64 1L));
  ignore (invoke options2 "[seasonal_d]" (C_int64 1L));
  ignore (invoke options2 "[seasonal_period]" (C_int64 12L));
  ignore (invoke options2 "[de_population_size]" (C_int64 100L));
  ignore (invoke options2 "[de_max_generations]" (C_int64 500L));
  let model2 = new_ARIMA (C_list [dv (List.rev !s); options2]) in
  let ar2 = fixdv (invoke model2 "ar_coefficients" C_void) in
  let sar2 = fixdv (invoke model2 "seasonal_ar_coefficients" C_void) in
  Printf.printf "AR coefficient: %g, seasonal AR coefficient: %g\n" (get_float (invoke ar2 "[]" (C_int 0))) (get_float (invoke sar2 "[]" (C_int 0)));
  Printf.printf "12-step forecast: %s\n" (str_of_dv (invoke model2 "forecast" (C_int64 12L)));

  print_endline "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================";
  let seasonal_shape = [| 0.8; 0.75; 0.9; 0.95; 1.0; 1.05; 1.1; 1.05; 1.0; 1.1; 1.3; 1.6 |] in
  Random.init 11;
  let y2 = ref [] in
  for i = 0 to 47 do
    let lvl = 100.0 +. 2.0 *. float_of_int i in
    y2 := (lvl *. seasonal_shape.(i mod 12) +. gauss () *. 3.0) :: !y2
  done;

  let es_options = new_ExponentialSmoothingOptions C_void in
  ignore (invoke es_options "[trend]" (C_int64 1L));
  ignore (invoke es_options "[seasonal]" (C_int64 2L));
  ignore (invoke es_options "[seasonal_period]" (C_int64 12L));
  ignore (invoke es_options "[de_population_size]" (C_int64 60L));
  ignore (invoke es_options "[de_max_generations]" (C_int64 300L));
  let es_model = new_ExponentialSmoothing (C_list [dv (List.rev !y2); es_options]) in
  Printf.printf "alpha=%g beta=%g gamma=%g\n" (get_float (invoke es_model "alpha" C_void)) (get_float (invoke es_model "beta" C_void)) (get_float (invoke es_model "gamma" C_void));
  Printf.printf "sigma^2: %g, AIC: %g\n" (get_float (invoke es_model "sigma2" C_void)) (get_float (invoke es_model "aic" C_void));
  Printf.printf "12-month forecast: %s\n" (str_of_dv (invoke es_model "forecast" (C_int64 12L)));

  print_endline "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================";
  let flat = dv [50.2; 49.8; 50.5; 49.6; 50.1; 50.3; 49.9; 50.0; 50.4; 49.7] in

  let ses = new_ExponentialSmoothing (C_list [flat; new_ExponentialSmoothingOptions C_void]) in
  Printf.printf "SES alpha= %g\n" (get_float (invoke ses "alpha" C_void));
  Printf.printf "SES 5-step forecast: %s\n" (str_of_dv (invoke ses "forecast" (C_int64 5L)));

  let holt_options = new_ExponentialSmoothingOptions C_void in
  ignore (invoke holt_options "[trend]" (C_int64 1L));
  let holt = new_ExponentialSmoothing (C_list [flat; holt_options]) in
  Printf.printf "Holt alpha=%g beta=%g\n" (get_float (invoke holt "alpha" C_void)) (get_float (invoke holt "beta" C_void));
  Printf.printf "Holt 5-step forecast: %s\n" (str_of_dv (invoke holt "forecast" (C_int64 5L)))
