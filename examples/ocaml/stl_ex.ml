open Swig
open Datamunge

let () =
  let n = 100 in
  let v = new_DVector C_void in
  for i = 0 to n - 1 do ignore (invoke v "push_back" (C_double (float_of_int i *. 1.5))) done;
  for i = 0 to n - 1 do Printf.printf "%g\n" (get_float (invoke v "[]" (C_int i))) done;

  let v2 = new_IVector C_void in
  for i = 0 to n - 1 do ignore (invoke v2 "push_back" (C_int (int_of_float (float_of_int i *. 1.5)))) done;
  for i = 0 to n - 1 do Printf.printf "%d\n" (get_int (invoke v2 "[]" (C_int i))) done;

  let p = new_IPair (C_list [C_int 3; C_int 4]) in
  Printf.printf "p: (%d, %d)\n" (get_int (invoke p "[first]" C_void)) (get_int (invoke p "[second]" C_void));

  let p2 = new_DPair (C_list [C_double 10.0; C_double 20.0]) in
  Printf.printf "p2: (%g, %g)\n" (get_float (invoke p2 "[first]" C_void)) (get_float (invoke p2 "[second]" C_void))
