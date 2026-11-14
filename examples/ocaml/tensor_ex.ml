(* CMD: just TARGET=tensor_ex run-ocaml *)
(* Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris. *)
open Datamunge

let shape_string tensor =
  "[" ^ String.concat ", " (List.map string_of_int (Tensor.shape tensor)) ^ "]"

let show = Tensor.to_string

let () =
  print_endline "=================== Construction ===================";
  Printf.printf "zeros([2,3]): %s\n" (show (Tensor.zeros [ 2; 3 ]));
  Printf.printf "eye(3): %s\n" (show (Tensor.eye 3));

  let r = Tensor.reshape (Tensor.arange ~step:1.0 0.0 12.0) [ 3; 4 ] in
  Printf.printf "arange(0,12).reshape([3,4]): %s\n" (show r);

  print_endline "\n=================== Shape ops ===================";
  Printf.printf "transpose -> shape %s\n" (shape_string (Tensor.transpose r));
  Printf.printf "slice(axis=1, start=1, stop=3): %s\n" (show (Tensor.slice r 1 1 3));

  print_endline "\n=================== Broadcasting arithmetic ===================";
  let col = Tensor.from_values [ 3; 1 ] [| 1.0; 2.0; 3.0 |] in
  let row = Tensor.from_values [ 1; 4 ] [| 10.0; 20.0; 30.0; 40.0 |] in
  Printf.printf "(3,1) + (1,4) -> %s\n" (show (Tensor.add col row));

  print_endline "\n=================== Reductions ===================";
  Printf.printf "r.sum() = %g, r.mean() = %g\n" (Tensor.sum r) (Tensor.mean r);
  Printf.printf "column means (axis=0): %s\n" (show (Tensor.mean_axis r 0));

  print_endline "\n=================== Linear algebra ===================";
  let a = Tensor.from_values [ 2; 3 ] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = Tensor.from_values [ 3; 2 ] [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |] in
  Printf.printf "matmul(2x3, 3x2) -> %s\n" (show (Tensor.matmul a b));
  let v1 = Tensor.from_values [ 3 ] [| 1.0; 2.0; 3.0 |] in
  let v2 = Tensor.from_values [ 3 ] [| 4.0; 5.0; 6.0 |] in
  Printf.printf "dot([1,2,3], [4,5,6]) = %g\n" (Tensor.dot v1 v2);
  Printf.printf "outer(v1, v2) -> %s\n" (show (Tensor.outer v1 v2));

  print_endline "\n=================== Comparisons & masks ===================";
  let mask = Tensor.greater_equal r (Tensor.full [ 3; 4 ] 6.0) in
  Printf.printf "r >= 6 -> %s\n" (show mask);
  Printf.printf "count(r >= 6) = %g\n" (Tensor.sum mask);

  print_endline "\n=================== A real dataset as a Tensor ===================";
  let iris = Data.iris () in
  let n = DataFrame.nrows iris in
  let columns = [| "Sepal.Length"; "Sepal.Width"; "Petal.Length"; "Petal.Width" |] in
  let flat = Array.init (n * 4) (fun k -> DataFrame.numeric_at iris columns.(k mod 4) (k / 4)) in
  let x = Tensor.from_values [ n; 4 ] flat in
  Printf.printf "iris feature tensor shape: %s\n" (shape_string x);

  let feature_means = Tensor.mean_axis x 0 in
  let centered = Tensor.subtract x (Tensor.reshape feature_means [ 1; 4 ]) in
  let scatter = Tensor.matmul (Tensor.transpose centered) centered in
  Printf.printf
    "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %s\n"
    (show feature_means);
  Printf.printf "(X-mean)^T (X-mean) [4x4 scatter matrix]: %s\n"
    (Tensor.to_string ~max_elements:16 scatter);

  let species = List.init n (DataFrame.string_at iris "Species") in
  let species_tensor = Tensor.from_string_values [ n ] species in
  let setosa_mask = Tensor.equal species_tensor (Tensor.from_string_values [ 1 ] [ "setosa" ]) in
  Printf.printf "setosa count = %g (of %d rows)\n" (Tensor.sum setosa_mask) n
