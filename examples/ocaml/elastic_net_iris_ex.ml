open Swig
open Datamunge

let formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n" (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));
  Printf.printf "formula: %s\n\n" formula;

  print_endline "=================== Ridge ===================";
  let ridge = new_Ridge (C_list [iris; C_string formula]) in
  ignore (invoke ridge "print_summary" C_void);

  print_endline "\n=================== Lasso ===================";
  let lasso = new_Lasso (C_list [iris; C_string formula]) in
  ignore (invoke lasso "print_summary" C_void);

  print_endline "\n================= Elastic Net =================";
  let elastic = new_ElasticNet (C_list [iris; C_string formula; C_double 0.5]) in
  ignore (invoke elastic "print_summary" C_void);

  print_endline "\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:";

  ignore (invoke (invoke ridge "plot_coefficient_path" C_void) "save" (C_string "elastic_net_ridge_path.svg"));
  ignore (invoke (invoke ridge "plot_cv_curve" C_void) "save" (C_string "elastic_net_ridge_cv.svg"));
  print_endline "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg";

  ignore (invoke (invoke lasso "plot_coefficient_path" C_void) "save" (C_string "elastic_net_lasso_path.svg"));
  ignore (invoke (invoke lasso "plot_cv_curve" C_void) "save" (C_string "elastic_net_lasso_cv.svg"));
  print_endline "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg";

  ignore (invoke (invoke elastic "plot_coefficient_path" C_void) "save" (C_string "elastic_net_elasticnet_path.svg"));
  ignore (invoke (invoke elastic "plot_cv_curve" C_void) "save" (C_string "elastic_net_elasticnet_cv.svg"));
  print_endline "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg"
