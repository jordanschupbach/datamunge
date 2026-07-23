(* optim_advanced_ex.ml -- exercises the 10 new datamunge::optim classes (ProximalGradient,
   FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP,
   InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from OCaml.

   HEADLINE FINDING (see this session's report for full detail): none of the 6 new pure-virtual
   interface types (ProximalFunction, HessianFunction, EqualityConstrainedFunction,
   InequalityConstrainedFunction, ResidualFunction, BayesianSurrogate) -- nor the pre-existing
   ArbitraryFunction that BayesianOptimization's objective argument needs even when using the
   ready-made RBFGaussianProcessSurrogate -- can be usefully subclassed from OCaml today. This
   reproduces and sharpens the project's already-documented OCaml director gap
   (memory/datamunge_ocaml_bindings.md):

     1. Directly invoking a pure-virtual director method (`invoke f "evaluate" coords`) reliably
        aborts the WHOLE PROCESS with an uncaught Swig::DirectorPureVirtualException (SIGABRT) --
        confirmed for both ArbitraryFunction::evaluate and DifferentiableFunction::gradient.
     2. When the director object is instead passed into a real C++ consumer (e.g. Newton calling
        f.gradient(x) internally), the callback DOES reach the OCaml dispatcher without crashing
        (confirmed via print tracing + an exact scalar double round-trip) -- but the incoming
        `const std::vector<double>&` coordinates argument arrives as a raw, unconvertible C_ptr:
        neither `invoke` nor this project's documented `create_std_..._from_ptr` workaround (which
        works for ordinary reference-returning methods elsewhere in this codebase) can turn it into
        a usable vector; both raise runtime failures. So even the non-crashing callback path cannot
        read its own input. Outgoing `std::vector<double>` RETURNS from the callback also do not
        appear to propagate back to C++ correctly (Newton's hessian() step was never reached
        across every trial, regardless of the gradient magnitude returned from OCaml).

   Net effect: constructing every one of the 10 classes (options structs + optimizer objects) and
   using the concrete, ready-made RBFGaussianProcessSurrogate directly (its fit()/acquisition() are
   ordinary forward calls INTO C++, not director callbacks, so they work fine) is fully usable from
   OCaml. Actually *running* any of the 10 optimizers to completion on a real, data-dependent
   custom objective is NOT usable from OCaml today, because every one of them requires a
   pure-virtual director subclass somewhere in the call graph. This file demonstrates both halves:
   the subclassing attempts (caught and reported, not crashing this program), and what does work. *)

open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let dvv rows =
  let vv = new_DVectorVector C_void in
  List.iter (fun row -> ignore (invoke vv "push_back" (dv row))) rows;
  vv

let list_of_dv v =
  let n = get_int (invoke v "size" C_void) in
  List.init n (fun i -> get_float (invoke v "[]" (C_int i)))

let () =
  print_endline "=== datamunge optim_advanced_ex (OCaml) ===\n";

  (* --- 2a: Newton on a 2D quadratic bowl via a HessianFunction director subclass. This is the
     critical test the project brief calls out: does directors/callback dispatch actually work
     for the new interface types? We do NOT invoke the pure-virtual methods directly (that is
     known to abort the whole process, see mini-test evidence in the report) -- instead we run
     it the "real" way, through Newton.optimize(), and report what actually happens. --- *)
  print_endline "--- Newton + HessianFunction (director subclass) ---";
  (let new_QuadBowl ob meth args =
     let coords = match args with C_list [ x ] -> x | x -> x in
     match meth with
     | "evaluate" ->
       (try
          let c = list_of_dv coords in
          match c with
          | [ x; y ] -> C_double (((x -. 1.0) ** 2.0) +. ((y -. 2.0) ** 2.0))
          | _ -> (invoke ob) meth args
        with _ -> C_double 0.0 (* incoming coords not readable, see report *))
     | "gradient" ->
       (try
          let c = list_of_dv coords in
          match c with
          | [ x; y ] -> dv [ 2.0 *. (x -. 1.0); 2.0 *. (y -. 2.0) ]
          | _ -> (invoke ob) meth args
        with _ -> dv [ 0.0; 0.0 ] (* incoming coords not readable, see report *))
     | "hessian" -> dvv [ [ 2.0; 0.0 ]; [ 0.0; 2.0 ] ]
     | _ -> (invoke ob) meth args
   in
   try
     let f = new_derived_object new_HessianFunction new_QuadBowl C_void in
     let coords = dv [ 0.0; 0.0 ] in
     let newton = new_Newton C_void in
     let final_val = get_float (invoke newton "optimize" (C_list [ f; coords ])) in
     Printf.printf
       "Newton.optimize() returned without crashing: f=%g, x=%s -- but the coordinates the \
        callback received were NOT readable (see report: incoming const vector<double>& arrives \
        as an unconvertible C_ptr), so this is NOT a real converged answer, just proof the call \
        completes.\n"
       final_val
       (String.concat ", " (List.map (Printf.sprintf "%g") (list_of_dv coords)))
   with e ->
     Printf.printf "Newton + HessianFunction director subclass FAILED: %s\n"
       (Printexc.to_string e));

  print_newline ();

  (* --- 2b: LevenbergMarquardt on a small nonlinear curve fit via a ResidualFunction director
     subclass. Same caveat as above applies -- included to confirm the same gap affects
     ResidualFunction, not just HessianFunction. --- *)
  print_endline "--- LevenbergMarquardt + ResidualFunction (director subclass) ---";
  (let ts = [ 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 ] in
   let true_a = 2.0 and true_k = 0.5 and true_c = 1.0 in
   let ys = List.map (fun t -> (true_a *. exp (-.true_k *. t)) +. true_c) ts in
   let new_ExpFit ob meth args =
     let coords = match args with C_list [ x ] -> x | x -> x in
     match meth with
     | "residuals" ->
       (try
          match list_of_dv coords with
          | [ a; k; c ] -> dv (List.map2 (fun t y -> (a *. exp (-.k *. t)) +. c -. y) ts ys)
          | _ -> (invoke ob) meth args
        with _ -> dv (List.map (fun _ -> 0.0) ts))
     | "jacobian" ->
       (try
          match list_of_dv coords with
          | [ _a; k; _c ] ->
            dvv (List.map (fun t -> [ exp (-.k *. t); -._a *. t *. exp (-.k *. t); 1.0 ]) ts)
          | _ -> (invoke ob) meth args
        with _ -> dvv (List.map (fun _ -> [ 0.0; 0.0; 0.0 ]) ts))
     | _ -> (invoke ob) meth args
   in
   try
     let f = new_derived_object new_ResidualFunction new_ExpFit C_void in
     let coords = dv [ 1.0; 1.0; 0.0 ] in
     let lm = new_LevenbergMarquardt C_void in
     let final_val = get_float (invoke lm "optimize" (C_list [ f; coords ])) in
     Printf.printf
       "LevenbergMarquardt.optimize() returned without crashing: cost=%g, params=%s -- same \
        caveat as Newton above: not a real fit, the callback could not read its own input.\n"
       final_val
       (String.concat ", " (List.map (Printf.sprintf "%g") (list_of_dv coords)))
   with e ->
     Printf.printf "LevenbergMarquardt + ResidualFunction director subclass FAILED: %s\n"
       (Printexc.to_string e));

  print_newline ();

  (* --- 2c: BayesianOptimization with the READY-TO-USE RBFGaussianProcessSurrogate (no
     subclassing needed for the surrogate itself). The objective function argument to optimize()
     is still an ArbitraryFunction, which needs the exact same director subclassing that just
     failed above -- confirming the "no subclassing needed" note in the brief applies only to
     the surrogate, not to the objective being optimized. --- *)
  print_endline
    "--- BayesianOptimization + RBFGaussianProcessSurrogate (concrete surrogate, objective \
     still needs a director) ---";
  (let new_Bowl1D ob meth args =
     let coords = match args with C_list [ x ] -> x | x -> x in
     match meth with
     | "evaluate" ->
       (try
          match list_of_dv coords with
          | [ x ] -> C_double (((x -. 3.0) ** 2.0) +. 1.0)
          | _ -> (invoke ob) meth args
        with _ -> C_double 0.0)
     | _ -> (invoke ob) meth args
   in
   try
     let f = new_derived_object new_ArbitraryFunction new_Bowl1D C_void in
     let coords = dv [ 0.0 ] in
     let lower = dv [ -5.0 ] in
     let upper = dv [ 5.0 ] in
     let surrogate = new_RBFGaussianProcessSurrogate C_void in
     let bo = new_BayesianOptimization C_void in
     let final_val =
       get_float (invoke bo "optimize" (C_list [ f; coords; lower; upper; surrogate ]))
     in
     Printf.printf "BayesianOptimization.optimize() returned without crashing: f=%g, x=%s\n"
       final_val
       (String.concat ", " (List.map (Printf.sprintf "%g") (list_of_dv coords)))
   with e ->
     Printf.printf
       "BayesianOptimization FAILED (objective still needs an ArbitraryFunction director \
        subclass, even though RBFGaussianProcessSurrogate itself needs no subclass): %s\n"
       (Printexc.to_string e));

  print_newline ();

  (* --- What IS fully usable from OCaml for all 10 classes without hitting the director gap:
     constructing/configuring their options structs and optimizer objects, and driving the
     concrete RBFGaussianProcessSurrogate directly (ordinary forward calls, no callback
     involved). --- *)
  print_endline "--- Confirmed-working: options structs, optimizer objects, concrete surrogate ---";
  let nopts = new_NewtonOptions C_void in
  ignore (invoke nopts "[max_iterations]" (C_int64 50L));
  Printf.printf "NewtonOptions.max_iterations set/get round-trips: %g\n"
    (get_float (invoke nopts "[max_iterations]" C_void));

  let lmopts = new_LevenbergMarquardtOptions C_void in
  ignore (invoke lmopts "[initial_damping]" (C_double 1e-2));
  Printf.printf "LevenbergMarquardtOptions.initial_damping set/get round-trips: %g\n"
    (get_float (invoke lmopts "[initial_damping]" C_void));

  ignore (new_ProximalGradient C_void);
  ignore (new_FISTA C_void);
  ignore (new_TrustRegionNewton C_void);
  ignore (new_AugmentedLagrangian C_void);
  ignore (new_SQP C_void);
  ignore (new_InteriorPoint C_void);
  print_endline
    "ProximalGradient/FISTA/TrustRegionNewton/AugmentedLagrangian/SQP/InteriorPoint all \
     construct fine with default options.";

  let surrogate = new_RBFGaussianProcessSurrogate (C_list [ C_double 1.0; C_double 1e-6 ]) in
  let pts = dvv [ [ 0.0 ]; [ 1.0 ]; [ 2.0 ] ] in
  let vals = dv [ 1.0; 0.0; 1.0 ] in
  ignore (invoke surrogate "fit" (C_list [ pts; vals ]));
  let acq = get_float (invoke surrogate "acquisition" (C_list [ dv [ 1.0 ]; C_double 0.0 ])) in
  Printf.printf
    "RBFGaussianProcessSurrogate constructed + fit() + acquisition() directly (ordinary forward \
     calls, no director involved): acquisition(1.0)=%g\n"
    acq;

  print_endline "\nOCaml exit"
