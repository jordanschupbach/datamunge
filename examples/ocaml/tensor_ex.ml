open Swig
open Datamunge

let sv values =
  let v = new_SizeVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_int64 (Int64.of_int x)))) values;
  v

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let strv values =
  let v = new_SVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_string x))) values;
  v

let shape_str t =
  let shp = invoke t "shape" C_void in
  let n = get_int (invoke shp "size" C_void) in
  String.concat ", " (List.init n (fun i -> string_of_int (get_int (invoke shp "[]" (C_int i)))))

let () =
  print_endline "=================== Construction ===================";
  let z = _Tensor_zeros (sv [2; 3]) in
  Printf.printf "zeros([2,3]): %s\n" (get_string (invoke z "to_string" C_void));

  let eyet = _Tensor_eye (C_int64 3L) in
  Printf.printf "eye(3): %s\n" (get_string (invoke eyet "to_string" C_void));

  let r = invoke (_Tensor_arange (C_list [C_double 0.0; C_double 12.0; C_double 1.0])) "reshape" (sv [3; 4]) in
  Printf.printf "arange(0,12).reshape([3,4]): %s\n" (get_string (invoke r "to_string" C_void));

  print_endline "\n=================== Shape ops ===================";
  let rt = invoke r "transpose" C_void in
  Printf.printf "transpose -> shape [%s]\n" (shape_str rt);
  let sliced = invoke r "slice" (C_list [C_int64 1L; C_int64 1L; C_int64 3L]) in
  Printf.printf "slice(axis=1, start=1, stop=3): %s\n" (get_string (invoke sliced "to_string" C_void));

  print_endline "\n=================== Broadcasting arithmetic ===================";
  let col = _Tensor_from_values (C_list [sv [3; 1]; dv [1.; 2.; 3.]]) in
  let row = _Tensor_from_values (C_list [sv [1; 4]; dv [10.; 20.; 30.; 40.]]) in
  let broadcast_sum = invoke col "add" row in
  Printf.printf "(3,1) + (1,4) -> %s\n" (get_string (invoke broadcast_sum "to_string" C_void));

  print_endline "\n=================== Reductions ===================";
  Printf.printf "r.sum() = %g, r.mean() = %g\n" (get_float (invoke r "sum" C_void)) (get_float (invoke r "mean" C_void));
  let col_means = invoke r "mean_axis" (C_int64 0L) in
  Printf.printf "column means (axis=0): %s\n" (get_string (invoke col_means "to_string" C_void));

  print_endline "\n=================== Linear algebra ===================";
  let a = _Tensor_from_values (C_list [sv [2; 3]; dv [1.; 2.; 3.; 4.; 5.; 6.]]) in
  let b = _Tensor_from_values (C_list [sv [3; 2]; dv [7.; 8.; 9.; 10.; 11.; 12.]]) in
  Printf.printf "matmul(2x3, 3x2) -> %s\n" (get_string (invoke (invoke a "matmul" b) "to_string" C_void));

  let v1 = _Tensor_from_values (C_list [sv [3]; dv [1.; 2.; 3.]]) in
  let v2 = _Tensor_from_values (C_list [sv [3]; dv [4.; 5.; 6.]]) in
  Printf.printf "dot([1,2,3], [4,5,6]) = %g\n" (get_float (invoke v1 "dot" v2));
  Printf.printf "outer(v1, v2) -> %s\n" (get_string (invoke (invoke v1 "outer" v2) "to_string" C_void));

  print_endline "\n=================== Comparisons & masks ===================";
  let mask = invoke r "greater_equal" (_Tensor_full (C_list [sv [3; 4]; C_double 6.0])) in
  Printf.printf "r >= 6 -> %s\n" (get_string (invoke mask "to_string" C_void));
  Printf.printf "count(r >= 6) = %g\n" (get_float (invoke mask "sum" C_void));

  print_endline "\n=================== A real dataset as a Tensor ===================";
  let iris = _DataFrame_iris C_void in
  let n = get_int (invoke iris "nrows" C_void) in
  let flat = ref [] in
  for i = 0 to n - 1 do
    let i64 = C_int64 (Int64.of_int i) in
    flat := !flat @ [
      get_float (invoke iris "numeric_at" (C_list [C_string "Sepal.Length"; i64]));
      get_float (invoke iris "numeric_at" (C_list [C_string "Sepal.Width"; i64]));
      get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Length"; i64]));
      get_float (invoke iris "numeric_at" (C_list [C_string "Petal.Width"; i64]))
    ]
  done;
  let x = _Tensor_from_values (C_list [sv [n; 4]; dv !flat]) in
  Printf.printf "iris feature tensor shape: [%s]\n" (shape_str x);

  let feature_means = invoke x "mean_axis" (C_int64 0L) in
  let centered = invoke x "subtract" (invoke feature_means "reshape" (sv [1; 4])) in
  let scatter = invoke (invoke centered "transpose" C_void) "matmul" centered in
  Printf.printf "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %s\n" (get_string (invoke feature_means "to_string" C_void));
  Printf.printf "(X-mean)^T (X-mean) [4x4 scatter matrix]: %s\n" (get_string (invoke scatter "to_string" (C_int64 16L)));

  let species = ref [] in
  for i = 0 to n - 1 do
    species := !species @ [get_string (invoke iris "string_at" (C_list [C_string "Species"; C_int64 (Int64.of_int i)]))]
  done;
  let species_tensor = _Tensor_from_string_values (C_list [sv [n]; strv !species]) in
  let setosa_mask = invoke species_tensor "equal" (_Tensor_from_string_values (C_list [sv [1]; strv ["setosa"]])) in
  Printf.printf "setosa count = %g (of %d rows)\n" (get_float (invoke setosa_mask "sum" C_void)) n
