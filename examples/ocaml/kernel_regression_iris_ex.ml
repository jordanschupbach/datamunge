(* CMD: just TARGET=kernel_regression_iris_ex run-ocaml *)
(* Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones. *)
open Datamunge

let formula = "Petal.Length ~ Petal.Width"

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n" (DataFrame.nrows iris) (DataFrame.ncols iris);
  Printf.printf "formula: %s\n\n" formula;

  (* Bandwidth auto-selected via leave-one-out CV. *)
  let model = KernelRegression.create iris formula in
  KernelRegression.print_summary model;

  Plot.save (KernelRegression.plot_fit model iris) "kernel_regression_iris_fit.svg";
  Plot.save (KernelRegression.plot_cv_curve model) "kernel_regression_iris_cv.svg";
  print_endline "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg";

  (* A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
     bias-variance story. *)
  let small = KernelRegression.create ~kernel:"gaussian" ~bandwidth:0.05 iris formula in
  Printf.printf "\nbandwidth=0.05 (too small): LOO R-squared=%g  LOO RMSE=%g\n"
    (KernelRegression.r_squared small) (KernelRegression.rmse small);
  Plot.save (KernelRegression.plot_fit small iris) "kernel_regression_iris_fit_small_bandwidth.svg";

  let large = KernelRegression.create ~kernel:"gaussian" ~bandwidth:5.0 iris formula in
  Printf.printf "bandwidth=5.0 (too large):  LOO R-squared=%g  LOO RMSE=%g\n"
    (KernelRegression.r_squared large) (KernelRegression.rmse large);
  Plot.save (KernelRegression.plot_fit large iris) "kernel_regression_iris_fit_large_bandwidth.svg";

  Printf.printf "bandwidth=%g (CV-selected): LOO R-squared=%g  LOO RMSE=%g\n"
    (KernelRegression.bandwidth model) (KernelRegression.r_squared model)
    (KernelRegression.rmse model);
  print_endline
    "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and \
     kernel_regression_iris_fit_large_bandwidth.svg"
