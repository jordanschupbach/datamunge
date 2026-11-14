(* CMD: just TARGET=stl_ex run-ocaml *)
(* C++ std::vector and std::pair values in OCaml: vectors are arrays or lists, pairs are tuples. *)
open Datamunge

let () =
  let n = 100 in
  let v = Array.init n (fun i -> float_of_int i *. 1.5) in
  Array.iter (Printf.printf "%g\n") v;
  let v2 = List.init n (fun i -> int_of_float (Float.floor (float_of_int i *. 1.5))) in
  List.iter (Printf.printf "%d\n") v2;

  let p = (3, 4) in
  Printf.printf "p: (%d, %d)\n" (fst p) (snd p);

  let p2 = (10, 20) in
  Printf.printf "p2: (%d, %d)\n" (fst p2) (snd p2);

  (* A C++ std::pair<double,double> comes back as a float * float tuple. *)
  let p3 = make_dpair 1.25 2.75 in
  let a, b = p3 in
  Printf.printf "make_dpair(1.25, 2.75): (%g, %g), sum_dpair = %g\n" a b (sum_dpair p3)
