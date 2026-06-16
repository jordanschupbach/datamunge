let () =
  let open Swig in
  ignore (Datamunge._hello C_void);
  print_endline "datamunge: _hello() ok";

  let p = Datamunge.new_DPair (C_list [ C_double 1.25; C_double 2.5 ]) in
  let p_first = get_float (invoke p "[first]" C_void) in
  let p_second = get_float (invoke p "[second]" C_void) in
  let p_sum = get_float (Datamunge._sum_dpair p) in
  Printf.printf "DPair: first=%g second=%g sum=%g\n" p_first p_second p_sum;

  let v = Datamunge.new_DVector C_void in
  ignore (invoke v "push_back" (C_double 3.0));
  ignore (invoke v "push_back" (C_double 4.5));
  let v0 = get_float (invoke v "[]" (C_int 0)) in
  let v1 = get_float (invoke v "[]" (C_int 1)) in
  let v_sum = get_float (Datamunge._sum_dvector v) in
  Printf.printf "DVector: [0]=%g [1]=%g sum=%g\n" v0 v1 v_sum
