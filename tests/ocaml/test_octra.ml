let test_import_and_hello () =
  ignore (Octra._hello Swig.C_void);
  Alcotest.(check pass) "_hello" () ()

let test_std_pair_and_vector () =
  let open Swig in
  let p = Octra.new_DPair (C_list [ C_double 1.25; C_double 2.5 ]) in
  let p_sum = get_float (Octra._sum_dpair p) in
  Alcotest.(check (float 1e-12)) "sum_dpair" 3.75 p_sum;

  let v = Octra.new_DVector C_void in
  ignore (invoke v "push_back" (C_double 3.0));
  ignore (invoke v "push_back" (C_double 4.5));
  let v_sum = get_float (Octra._sum_dvector v) in
  Alcotest.(check (float 1e-12)) "sum_dvector" 7.5 v_sum

let () =
  Alcotest.run "octra-ocaml"
    [
      ( "basic",
        [
          Alcotest.test_case "import+hello" `Quick test_import_and_hello;
          Alcotest.test_case "std pair+vector" `Quick test_std_pair_and_vector;
        ] );
    ]
