(* CMD: just TARGET=datamunge_ex run-ocaml *)
(* Smoke test of the basic bindings: vectors, pairs and a native Callback object. *)
open Datamunge

let () =
  hello ();
  let v = make_dvector 1.0 2.0 3.0 in
  Printf.printf "sum_dvector:\t%g\n" (sum_dvector v);
  let p = make_dpair 1.25 2.75 in
  Printf.printf "sum_dpair:\t%g\n" (sum_dpair p);
  let cb = Callback.create () in
  Printf.printf "call_with_callback(3.0):\t%g\n" (call_with_callback 3.0 cb);
  let v2 = map_dvector_with_callback (make_dvector 1.0 2.0 3.0) cb in
  Printf.printf "sum_dvector(map_dvector_with_callback(1,2,3)):\t%g\n" (sum_dvector v2);
  Callback.close cb
