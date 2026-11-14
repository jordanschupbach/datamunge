(* CMD: just TARGET=ggplot_ex run-ocaml *)
(* Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library. *)
open Datamunge

let () =
  let iris = Data.iris () in

  (* Each GGPlot method returns the plot, so a ggplot2 "+ layer" chain becomes a pipeline. *)

  (* ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point() *)
  GGPlot.create ~y_column:"Sepal.Width" ~color_column:"Species" iris "Sepal.Length"
  |> GGPlot.geom_point
  |> GGPlot.labs ~title:"Iris Sepal Dimensions" ~x:"Sepal Length" ~y:"Sepal Width"
  |> GGPlot.theme_minimal
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_point.svg";

  (* + geom_smooth(): an lm() fit line, reusing stats::LM internally. *)
  GGPlot.create ~y_column:"Petal.Length" iris "Sepal.Length"
  |> GGPlot.geom_point ~color:(RGB.create_int_int_int_float 156 163 175) ~size:2.5
  |> GGPlot.geom_smooth
  |> GGPlot.labs ~title:"Petal Length vs Sepal Length With a Linear Fit" ~x:"Sepal Length"
       ~y:"Petal Length"
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_smooth.svg";

  (* geom_bar(): counts a discrete column (stat = "count"). *)
  GGPlot.create iris "Species"
  |> GGPlot.geom_bar
  |> GGPlot.labs ~title:"Observations per Species" ~x:"Species" ~y:"Count"
  |> GGPlot.theme_bw
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_bar.svg";

  (* geom_boxplot(): grouped by a discrete x column. *)
  GGPlot.create ~y_column:"Petal.Width" iris "Species"
  |> GGPlot.geom_boxplot
  |> GGPlot.labs ~title:"Petal Width by Species" ~x:"Species" ~y:"Petal Width"
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_boxplot.svg";

  (* geom_histogram() + geom_density(): distribution of a single numeric column. *)
  GGPlot.create iris "Sepal.Length"
  |> GGPlot.geom_histogram ~bins:20
  |> GGPlot.labs ~title:"Distribution of Sepal Length" ~x:"Sepal Length" ~y:"Count"
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_histogram.svg";

  GGPlot.create iris "Sepal.Length"
  |> GGPlot.geom_density
  |> GGPlot.labs ~title:"Density of Sepal Length" ~x:"Sepal Length" ~y:"Density"
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_density.svg";

  (* facet_wrap(): one panel per Species, composed via RLayout under the hood. *)
  GGPlot.create ~y_column:"Petal.Width" iris "Petal.Length"
  |> GGPlot.geom_point
  |> (fun p -> GGPlot.facet_wrap p "Species")
  |> GGPlot.labs ~title:"Petal Dimensions" ~x:"Petal Length" ~y:"Petal Width"
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_facet.svg";

  (* theme_classic(): another built-in theme. *)
  GGPlot.create ~y_column:"Sepal.Width" ~color_column:"Species" iris "Sepal.Length"
  |> GGPlot.geom_point
  |> GGPlot.theme_classic
  |> fun p -> GGPlot.save_svg p "ocaml_ggplot_classic.svg";

  print_endline "Wrote 7 SVGs to ocaml_ggplot_*.svg"
