(* CMD: just TARGET=rplot_ex run-ocaml *)
(* R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts. *)
open Datamunge

let () =
  (* plot(x, y, type = "p") then abline() layered on afterward. *)
  let scatter =
    RPlot.plot ~type_:"p" ~label:"observed" [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
      [| 2.1; 3.9; 6.2; 7.8; 10.1 |]
  in
  let red = RGB.create_int_int_int_float 220 38 38 in
  ignore (RPlot.abline ~color:red ~stroke_width:1.5 scatter 0.0 2.0);
  ignore (RPlot.title scatter "plot() + abline()");
  ignore (RPlot.x_label scatter "x");
  ignore (RPlot.y_label scatter "y");
  RPlot.save_svg scatter "ocaml_rplot_scatter.svg";

  (* hist(): equal-width binning over the data range. *)
  let histogram =
    RPlot.hist ~bins:6 ~label:"counts"
      [| 1.0; 2.; 2.; 3.; 3.; 3.; 4.; 4.; 4.; 4.; 5.; 5.; 5.; 6.; 6.; 7. |]
  in
  RPlot.save_svg histogram "ocaml_rplot_hist.svg";

  (* barplot(): categorical positions with x tick labels. *)
  RPlot.save_svg
    (RPlot.title (RPlot.barplot ~names:[ "A"; "B"; "C" ] [| 23.0; 41.0; 12.0 |]) "barplot()")
    "ocaml_rplot_barplot.svg";

  (* boxplot(): Tukey five-number summary per group. *)
  let groups =
    [ [| 2.0; 4.; 4.; 4.; 5.; 5.; 7.; 9. |]; [| 1.0; 2.; 2.; 2.; 3.; 3.; 3.; 3.; 4.; 20. |] ]
  in
  RPlot.save_svg
    (RPlot.title (RPlot.boxplot ~names:[ "low variance"; "has outlier" ] groups) "boxplot()")
    "ocaml_rplot_boxplot.svg";

  (* pie(): wedge areas proportional to value, axes hidden automatically. *)
  RPlot.save_svg
    (RPlot.title (RPlot.pie ~names:[ "Q1"; "Q2"; "Q3"; "Q4" ] [| 35.0; 25.0; 20.0; 20.0 |]) "pie()")
    "ocaml_rplot_pie.svg";

  (* curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback). *)

  (* qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line. *)
  let residuals = [| -2.1; -1.3; -0.8; -0.4; -0.1; 0.2; 0.5; 0.9; 1.4; 2.3 |] in
  RPlot.save_svg (RPlot.qqline (RPlot.qqnorm residuals) residuals) "ocaml_rplot_qqnorm.svg";

  (* par(mfrow = c(1, 2))-style multi-panel composition via RLayout. *)
  let layout = RLayout.add (RLayout.add (RLayout.create_ 1 2) scatter) histogram in
  RLayout.save_svg layout "ocaml_rplot_layout.svg";

  print_endline "Wrote 7 SVGs to ocaml_rplot_*.svg"
