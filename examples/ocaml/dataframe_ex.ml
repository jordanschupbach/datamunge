(* See lm_ex.ml for notes on this binding's conventions: invoke obj "method" args for
   methods, invoke obj "[field]" args for struct fields, invoke v "[]" (C_int i) for
   container indexing, and C_int64 (not C_int) for any std::size_t-typed argument. *)
open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let sv values =
  let v = new_SVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_string x))) values;
  v

let iv values =
  let v = new_IVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_int x))) values;
  v

let () =
  let sales = new_DataFrame C_void in
  ignore (invoke sales "add_string_column" (C_list [C_string "region"; sv ["west"; "west"; "east"; "south"; "south"; "south"]]));
  ignore (invoke sales "add_string_column" (C_list [C_string "product"; sv ["widget"; "widget"; "widget"; "gizmo"; "gizmo"; "gizmo"]]));
  ignore (invoke sales "add_numeric_column" (C_list [C_string "sales"; dv [10.; 10.; 14.; 8.; 0.; 11.]; iv [1; 1; 1; 1; 0; 1]]));
  ignore (invoke sales "add_string_column" (C_list [C_string "quarter"; sv ["Q1"; "Q1"; "Q1"; "Q2"; "Q2"; ""]; iv [1; 1; 1; 1; 1; 0]]));

  print_endline "raw data";
  print_endline (get_string (invoke sales "to_string" C_void));
  print_newline ();

  let cleaned = invoke sales "drop_duplicates" (sv ["region"; "product"; "sales"; "quarter"]) in
  ignore (invoke cleaned "fill_null_string" (C_list [C_string "quarter"; C_string "unknown"]));
  ignore (invoke cleaned "fill_null_numeric" (C_list [C_string "sales"; C_double 0.0]));
  print_endline "after drop_duplicates + fill_null";
  print_endline (get_string (invoke cleaned "to_string" C_void));
  print_newline ();

  let selected_pre = invoke cleaned "select" (sv ["region"; "sales"; "quarter"]) in
  let selected = invoke selected_pre "sort_by" (C_list [C_string "sales"; C_bool false]) in
  print_endline "selected + sorted";
  print_endline (get_string (invoke selected "to_string" C_void));
  print_newline ();

  let grouped_pre = invoke cleaned "group_by_sum" (C_list [sv ["region"]; sv ["sales"]]) in
  let grouped = invoke grouped_pre "sort_by" (C_list [C_string "sales"; C_bool false]) in
  print_endline "group_by_sum(region)";
  print_endline (get_string (invoke grouped "to_string" C_void));
  print_newline ();

  let targets = new_DataFrame C_void in
  ignore (invoke targets "add_string_column" (C_list [C_string "region"; sv ["west"; "east"; "south"]]));
  ignore (invoke targets "add_numeric_column" (C_list [C_string "target"; dv [18.; 12.; 25.]]));
  let joined = invoke grouped "join" (C_list [targets; C_string "region"; C_string "region"; C_bool true]) in
  print_endline "joined with targets";
  print_endline (get_string (invoke joined "to_string" C_void));
  print_newline ();

  let shape = invoke cleaned "shape" C_void in
  Printf.printf "shape = (%d, %d)\n"
    (get_int (invoke shape "[]" (C_int 0)))
    (get_int (invoke shape "[]" (C_int 1)));
  Printf.printf "sales count = %g\n" (get_float (invoke cleaned "numeric_count" (C_string "sales")));
  Printf.printf "sales nulls = %g\n" (get_float (invoke cleaned "numeric_null_count" (C_string "sales")));
  Printf.printf "sales sum = %g\n" (get_float (invoke cleaned "numeric_sum" (C_string "sales")));
  Printf.printf "sales mean = %g\n" (get_float (invoke cleaned "numeric_mean" (C_string "sales")))
