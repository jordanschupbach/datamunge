(* CMD: just TARGET=timeseries_ex run-ocaml *)
(* Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters). *)
open Datamunge

let pi = Float.pi

(* A standard normal draw by Box-Muller on (0, 1] uniforms (avoids log(0)); reseed with
   Random.init. *)
let gauss () =
  let u1 = 1. -. Random.float 1. in
  let u2 = 1. -. Random.float 1. in
  sqrt (-2. *. log u1) *. cos (2. *. pi *. u2)

let joined values = String.concat ", " (Array.to_list (Array.map (Printf.sprintf "%g") values))

let () =
  print_endline
    "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================";
  Random.init 42;
  let level = ref 100.0 in
  let prev_shock = ref 0.0 in
  let y =
    Array.init 150 (fun _ ->
        let shock = gauss () in
        (* integrated ARMA(0,1) increments with drift *)
        level := !level +. 0.3 +. (0.4 *. !prev_shock) +. shock;
        prev_shock := shock;
        !level)
  in
  let options = ARIMAOptions.create () in
  ARIMAOptions.set_p options 1;
  ARIMAOptions.set_d options 1;
  ARIMAOptions.set_q options 1;
  ARIMAOptions.set_de_population_size options 80;
  ARIMAOptions.set_de_max_generations options 400;
  let model = ARIMA.create ~options y in

  Printf.printf "AR coefficient: %g\n" (ARIMA.ar_coefficients model).(0);
  Printf.printf "MA coefficient: %g\n" (ARIMA.ma_coefficients model).(0);
  Printf.printf "sigma^2: %g, AIC: %g, BIC: %g\n" (ARIMA.sigma2 model) (ARIMA.aic model)
    (ARIMA.bic model);

  let forecast_values, std_errors = ARIMA.forecast_with_intervals model 6 in
  Printf.printf "6-step forecast: %s\n" (joined forecast_values);
  Printf.printf "forecast std. errors: %s\n" (joined std_errors);

  print_endline
    "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================";
  Random.init 7;
  let prev = ref 0.0 in
  let s =
    Array.init 120 (fun i ->
        let i = float_of_int i in
        prev := (0.5 *. !prev) +. gauss ();
        20. +. (0.2 *. i) +. (5. *. sin (2. *. pi *. i /. 12.)) +. !prev)
  in
  let options2 = ARIMAOptions.create () in
  ARIMAOptions.set_p options2 1;
  ARIMAOptions.set_seasonal_p options2 1;
  ARIMAOptions.set_seasonal_d options2 1;
  ARIMAOptions.set_seasonal_period options2 12;
  ARIMAOptions.set_de_population_size options2 100;
  ARIMAOptions.set_de_max_generations options2 500;
  let model2 = ARIMA.create ~options:options2 s in
  Printf.printf "AR coefficient: %g, seasonal AR coefficient: %g\n"
    (ARIMA.ar_coefficients model2).(0)
    (ARIMA.seasonal_ar_coefficients model2).(0);
  Printf.printf "12-step forecast: %s\n" (joined (ARIMA.forecast model2 12));

  print_endline
    "\n\
     =================== Exponential smoothing: Holt-Winters on retail-style seasonal data \
     ===================";
  let seasonal_shape = [| 0.8; 0.75; 0.9; 0.95; 1.0; 1.05; 1.1; 1.05; 1.0; 1.1; 1.3; 1.6 |] in
  Random.init 11;
  let y2 =
    Array.init 48 (fun i ->
        ((100. +. (2. *. float_of_int i)) *. seasonal_shape.(i mod 12)) +. (gauss () *. 3.))
  in
  let es_options = ExponentialSmoothingOptions.create () in
  ExponentialSmoothingOptions.set_trend es_options TrendType.Additive;
  ExponentialSmoothingOptions.set_seasonal es_options SeasonalType.Multiplicative;
  ExponentialSmoothingOptions.set_seasonal_period es_options 12;
  ExponentialSmoothingOptions.set_de_population_size es_options 60;
  ExponentialSmoothingOptions.set_de_max_generations es_options 300;
  let es_model = ExponentialSmoothing.create ~options:es_options y2 in
  Printf.printf "alpha=%g beta=%g gamma=%g\n" (ExponentialSmoothing.alpha es_model)
    (ExponentialSmoothing.beta es_model)
    (ExponentialSmoothing.gamma es_model);
  Printf.printf "sigma^2: %g, AIC: %g\n" (ExponentialSmoothing.sigma2 es_model)
    (ExponentialSmoothing.aic es_model);
  Printf.printf "12-month forecast: %s\n" (joined (ExponentialSmoothing.forecast es_model 12));

  print_endline
    "\n\
     =================== Simple exponential smoothing vs. Holt's linear trend \
     ===================";
  let flat = [| 50.2; 49.8; 50.5; 49.6; 50.1; 50.3; 49.9; 50.0; 50.4; 49.7 |] in

  let ses = ExponentialSmoothing.create ~options:(ExponentialSmoothingOptions.create ()) flat in
  Printf.printf "SES alpha= %g\n" (ExponentialSmoothing.alpha ses);
  Printf.printf "SES 5-step forecast: %s\n" (joined (ExponentialSmoothing.forecast ses 5));

  let holt_options = ExponentialSmoothingOptions.create () in
  ExponentialSmoothingOptions.set_trend holt_options TrendType.Additive;
  let holt = ExponentialSmoothing.create ~options:holt_options flat in
  Printf.printf "Holt alpha=%g beta=%g\n" (ExponentialSmoothing.alpha holt)
    (ExponentialSmoothing.beta holt);
  Printf.printf "Holt 5-step forecast: %s\n" (joined (ExponentialSmoothing.forecast holt 5))
