open Swig
open Datamunge

let rosenbrock_dual x0 x1 =
  let a = invoke (new_Dual (C_list [C_double 1.0; C_double 0.0])) "subtract" x0 in
  let b = invoke x1 "subtract" (invoke x0 "multiply" x0) in
  invoke (invoke a "multiply" a) "add" (invoke (invoke b "multiply" b) "multiply_scalar" (C_double 100.0))

let rosenbrock_value x0 x1 =
  let a = 1.0 -. x0 in
  let b = x1 -. x0 *. x0 in
  a *. a +. 100.0 *. b *. b

let () =
  print_endline "=================== Forward mode: scalar derivative ===================";
  let x0 = new_Dual (C_list [C_double 2.0; C_double 1.0]) in
  let f = invoke (invoke (invoke x0 "multiply" x0) "multiply" x0) "subtract" (invoke x0 "multiply_scalar" (C_double 2.0)) in
  Printf.printf "f(x) = x^3 - 2x, f'(2) = %g (exact: 10)\n" (get_float (invoke f "derivative" C_void));

  print_endline "\n=================== Reverse mode: build a graph by hand ===================";
  let tape = new_Tape C_void in
  let a = new_Var (C_list [tape; C_double 2.0]) in
  let b = new_Var (C_list [tape; C_double 3.0]) in
  let y = invoke (invoke a "multiply" b) "add" (invoke a "sin" C_void) in
  Printf.printf "y = a*b + sin(a) at a=2, b=3 -> y = %g\n" (get_float (invoke y "value" C_void));
  let adjoint = invoke tape "backward" y in
  Printf.printf "dy/da = %g (exact: b + cos(a))\n" (get_float (invoke adjoint "[]" (C_int (get_int (invoke a "index" C_void)))));
  Printf.printf "dy/db = %g (exact: a)\n" (get_float (invoke adjoint "[]" (C_int (get_int (invoke b "index" C_void)))));

  print_endline "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================";
  let px = 0.0 and py = 0.0 in

  let gx = get_float (invoke (rosenbrock_dual (new_Dual (C_list [C_double px; C_double 1.0])) (new_Dual (C_list [C_double py; C_double 0.0]))) "derivative" C_void) in
  let gy = get_float (invoke (rosenbrock_dual (new_Dual (C_list [C_double px; C_double 0.0])) (new_Dual (C_list [C_double py; C_double 1.0]))) "derivative" C_void) in

  let tape2 = new_Tape C_void in
  let vx = new_Var (C_list [tape2; C_double px]) in
  let vy = new_Var (C_list [tape2; C_double py]) in
  let va = invoke (new_Var (C_list [tape2; C_double 1.0])) "subtract" vx in
  let vb = invoke vy "subtract" (invoke vx "multiply" vx) in
  let vf = invoke (invoke va "multiply" va) "add" (invoke (invoke vb "multiply" vb) "multiply_scalar" (C_double 100.0)) in
  let grad_rev = invoke tape2 "backward" vf in

  Printf.printf "f(0,0) = %g\n" (rosenbrock_value px py);
  Printf.printf "gradient (forward mode): [%g, %g]\n" gx gy;
  Printf.printf "gradient (reverse mode): [%g, %g]\n"
    (get_float (invoke grad_rev "[]" (C_int (get_int (invoke vx "index" C_void)))))
    (get_float (invoke grad_rev "[]" (C_int (get_int (invoke vy "index" C_void)))));

  print_endline "\n=================== Jacobian of a vector-valued function ===================";
  let vx_val = 2.0 and vy_val = 3.0 in
  let jac = Array.make_matrix 3 2 0.0 in
  let seeds = [| (0, 1.0, 0.0); (1, 0.0, 1.0) |] in
  Array.iter (fun (col, sx, sy) ->
    let dx = new_Dual (C_list [C_double vx_val; C_double sx]) in
    let dy = new_Dual (C_list [C_double vy_val; C_double sy]) in
    let ox = invoke dx "multiply" dx in
    let oy = invoke dx "multiply" dy in
    let oz = invoke (invoke dy "multiply" dy) "multiply" dy in
    jac.(0).(col) <- get_float (invoke ox "derivative" C_void);
    jac.(1).(col) <- get_float (invoke oy "derivative" C_void);
    jac.(2).(col) <- get_float (invoke oz "derivative" C_void)
  ) seeds;
  print_endline "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:";
  Array.iter (fun row -> Printf.printf "  [%g, %g]\n" row.(0) row.(1)) jac;

  print_endline "\n=================== Hessian via second-order forward mode (HyperDual) ===================";
  let mx = 1.0 and my = 1.0 in
  let rosenbrock_hyperdual hx0 hx1 =
    let a = invoke (new_HyperDual (C_list [C_double 1.0; C_double 0.0; C_double 0.0; C_double 0.0])) "subtract" hx0 in
    let b = invoke hx1 "subtract" (invoke hx0 "multiply" hx0) in
    invoke (invoke a "multiply" a) "add" (invoke (invoke b "multiply" b) "multiply_scalar" (C_double 100.0))
  in
  let hseeds = [| (1.0, 0.0); (0.0, 1.0) |] in
  let hmat = Array.make_matrix 2 2 0.0 in
  for i = 0 to 1 do
    for j = 0 to 1 do
      let (e1x, e1y) = hseeds.(i) and (e2x, e2y) = hseeds.(j) in
      let hx = new_HyperDual (C_list [C_double mx; C_double e1x; C_double e2x; C_double 0.0]) in
      let hy = new_HyperDual (C_list [C_double my; C_double e1y; C_double e2y; C_double 0.0]) in
      hmat.(i).(j) <- get_float (invoke (rosenbrock_hyperdual hx hy) "eps1eps2" C_void)
    done
  done;
  print_endline "Hessian of the Rosenbrock function at its minimum (1,1):";
  Array.iter (fun row -> Printf.printf "  [%g, %g]\n" row.(0) row.(1)) hmat;

  print_endline "\n=================== Gradient descent driven by reverse-mode gradients ===================";
  let point = [| -1.2; 1.0 |] in
  let learning_rate = 0.001 in
  let n_steps = 2000 in
  for step = 0 to n_steps - 1 do
    let t = new_Tape C_void in
    let vx2 = new_Var (C_list [t; C_double point.(0)]) in
    let vy2 = new_Var (C_list [t; C_double point.(1)]) in
    let va2 = invoke (new_Var (C_list [t; C_double 1.0])) "subtract" vx2 in
    let vb2 = invoke vy2 "subtract" (invoke vx2 "multiply" vx2) in
    let vf2 = invoke (invoke va2 "multiply" va2) "add" (invoke (invoke vb2 "multiply" vb2) "multiply_scalar" (C_double 100.0)) in
    let grad = invoke t "backward" vf2 in
    let loss = get_float (invoke vf2 "value" C_void) in
    point.(0) <- point.(0) -. learning_rate *. get_float (invoke grad "[]" (C_int (get_int (invoke vx2 "index" C_void))));
    point.(1) <- point.(1) -. learning_rate *. get_float (invoke grad "[]" (C_int (get_int (invoke vy2 "index" C_void))));
    if step = 0 || step = n_steps - 1 then
      Printf.printf "step %d: loss = %g, x = [%g, %g]\n" step loss point.(0) point.(1)
  done;
  print_endline "(true minimum is at [1, 1] with loss 0)"
