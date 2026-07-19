open Swig
open Datamunge

let formula = "Petal.Length ~ Petal.Width"

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n" (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));
  Printf.printf "formula: %s\n\n" formula;

  let model = new_KernelRegression (C_list [iris; C_string formula]) in
  ignore (invoke model "print_summary" C_void);

  ignore (invoke (invoke model "plot_fit" iris) "save" (C_string "kernel_regression_iris_fit.svg"));
  ignore (invoke (invoke model "plot_cv_curve" C_void) "save" (C_string "kernel_regression_iris_cv.svg"));
  print_endline "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg";

  let small = new_KernelRegression (C_list [iris; C_string formula; C_string "gaussian"; C_double 0.05]) in
  Printf.printf "\nbandwidth=0.05 (too small): LOO R-squared=%g  LOO RMSE=%g\n"
    (get_float (invoke small "r_squared" C_void)) (get_float (invoke small "rmse" C_void));
  ignore (invoke (invoke small "plot_fit" iris) "save" (C_string "kernel_regression_iris_fit_small_bandwidth.svg"));

  let large = new_KernelRegression (C_list [iris; C_string formula; C_string "gaussian"; C_double 5.0]) in
  Printf.printf "bandwidth=5.0 (too large):  LOO R-squared=%g  LOO RMSE=%g\n"
    (get_float (invoke large "r_squared" C_void)) (get_float (invoke large "rmse" C_void));
  ignore (invoke (invoke large "plot_fit" iris) "save" (C_string "kernel_regression_iris_fit_large_bandwidth.svg"));

  Printf.printf "bandwidth=%g (CV-selected): LOO R-squared=%g  LOO RMSE=%g\n"
    (get_float (invoke model "bandwidth" C_void)) (get_float (invoke model "r_squared" C_void)) (get_float (invoke model "rmse" C_void));
  print_endline "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg"
