(* CMD: just TARGET=elastic_net_iris_ex run-ocaml *)
(* Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures. *)

open Datamunge

let formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

let () =
  let iris = Data.iris () in
  Printf.printf "iris: %d rows x %d cols\n" (DataFrame.nrows iris) (DataFrame.ncols iris);
  Printf.printf "formula: %s\n\n" formula;

  print_endline "=================== Ridge ===================";
  let ridge = Ridge.create iris formula in
  Ridge.print_summary ridge;

  print_endline "\n=================== Lasso ===================";
  let lasso = Lasso.create iris formula in
  Lasso.print_summary lasso;

  print_endline "\n================= Elastic Net =================";
  let elastic = ElasticNet.create ~alpha:0.5 iris formula in
  ElasticNet.print_summary elastic;

  print_endline
    "\n\
     Saved figures showing how each model's coefficients respond to the regularization strength, \
     and the cross-validation curve used to pick it:";
  (* The three model types share no base class, so pair each one's two plots up front. *)
  List.iter
    (fun (name, label, path_plot, cv_plot) ->
      let path_file = Printf.sprintf "elastic_net_%s_path.svg" name in
      let cv_file = Printf.sprintf "elastic_net_%s_cv.svg" name in
      RPlot.save path_plot path_file;
      RPlot.save cv_plot cv_file;
      Printf.printf "  %s %s, %s\n" label path_file cv_file)
    [
      ("ridge", "ridge:      ", Ridge.plot_coefficient_path ridge, Ridge.plot_cv_curve ridge);
      ("lasso", "lasso:      ", Lasso.plot_coefficient_path lasso, Lasso.plot_cv_curve lasso);
      ( "elasticnet",
        "elastic net:",
        ElasticNet.plot_coefficient_path elastic,
        ElasticNet.plot_cv_curve elastic );
    ]
