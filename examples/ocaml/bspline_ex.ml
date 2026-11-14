(* CMD: just TARGET=bspline_ex run-ocaml *)
(* Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict. *)

open Datamunge

let () =
  (* Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2. *)
  let grid = List.init 8 (fun iy -> List.init 8 (fun ix -> (float ix /. 7.0, float iy /. 7.0))) in
  let points = Array.of_list (List.concat grid) in
  let data = DataFrame.create () in
  DataFrame.add_numeric_column data "x" (Array.map fst points);
  DataFrame.add_numeric_column data "y" (Array.map snd points);
  DataFrame.add_numeric_column data "z" (Array.map (fun (x, y) -> sin x +. (y *. y)) points);

  let surface = LM.create data "z ~ bs(x, y)" in
  print_endline "Fitted z ~ bs(x, y)";
  LM.print_summary surface;

  let new_points = DataFrame.create () in
  DataFrame.add_numeric_column new_points "x" [| 0.25; 0.75 |];
  DataFrame.add_numeric_column new_points "y" [| 0.50; 0.25 |];
  let pred = LM.predict surface new_points in
  Printf.printf "Predictions: [%.6f, %.6f]\n" pred.(0) pred.(1)
