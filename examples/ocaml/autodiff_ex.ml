(* CMD: just TARGET=autodiff_ex run-ocaml *)
(* Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual). *)

(* Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
   (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be bound.
   This port demonstrates the same ideas with the bound named-method API. [Var.index v] is the
   Var's position in the adjoint array [Tape.backward tape output] returns. *)

open Datamunge

let rosenbrock_dual x0 x1 =
  let a = Dual.subtract (Dual.create ~derivative:0.0 1.0) x0 in
  let b = Dual.subtract x1 (Dual.multiply x0 x0) in
  Dual.add (Dual.multiply a a) (Dual.multiply_scalar (Dual.multiply b b) 100.0)

let rosenbrock_value x0 x1 = ((1.0 -. x0) ** 2.0) +. (100.0 *. ((x1 -. (x0 *. x0)) ** 2.0))

(* Build the Rosenbrock graph on tape; returns the output Var. *)
let rosenbrock_var tape x0 x1 =
  let a = Var.subtract (Var.create tape 1.0) x0 in
  let b = Var.subtract x1 (Var.multiply x0 x0) in
  Var.add (Var.multiply a a) (Var.multiply_scalar (Var.multiply b b) 100.0)

let rosenbrock_hyper_dual x0 x1 =
  let a = HyperDual.subtract (HyperDual.create 1.0) x0 in
  let b = HyperDual.subtract x1 (HyperDual.multiply x0 x0) in
  HyperDual.add (HyperDual.multiply a a) (HyperDual.multiply_scalar (HyperDual.multiply b b) 100.0)

let adjoint adjoints v = adjoints.(Var.index v)

let print_row row =
  Printf.printf "  [%s]\n" (String.concat ", " (List.map (Printf.sprintf "%g") row))

let () =
  print_endline "=================== Forward mode: scalar derivative ===================";
  let x0 = Dual.create ~derivative:1.0 2.0 (* seed derivative = 1 to read df/dx directly *) in
  let f = Dual.subtract (Dual.multiply (Dual.multiply x0 x0) x0) (Dual.multiply_scalar x0 2.0) in
  Printf.printf "f(x) = x^3 - 2x, f'(2) = %g (exact: 10)\n" (Dual.derivative f);

  print_endline "\n=================== Reverse mode: build a graph by hand ===================";
  let tape = Tape.create () in
  let a = Var.create tape 2.0 in
  let b = Var.create tape 3.0 in
  let y = Var.add (Var.multiply a b) (Var.sin a) in
  Printf.printf "y = a*b + sin(a) at a=2, b=3 -> y = %g\n" (Var.value y);
  let adj = Tape.backward tape y in
  Printf.printf "dy/da = %g (exact: b + cos(a))\n" (adjoint adj a);
  Printf.printf "dy/db = %g (exact: a)\n" (adjoint adj b);

  print_endline
    "\n\
     =================== Forward vs reverse mode agree on the Rosenbrock function \
     ===================";
  let px = 0.0 and py = 0.0 in
  (* Forward mode: one Dual pass per partial derivative, seeding the direction of interest. *)
  let partial sx sy =
    Dual.derivative
      (rosenbrock_dual (Dual.create ~derivative:sx px) (Dual.create ~derivative:sy py))
  in
  let gx = partial 1.0 0.0 and gy = partial 0.0 1.0 in
  (* Reverse mode: one Tape/Var pass computes every partial at once. *)
  let tape = Tape.create () in
  let vx = Var.create tape px in
  let vy = Var.create tape py in
  let grad = Tape.backward tape (rosenbrock_var tape vx vy) in
  Printf.printf "f(0,0) = %g\n" (rosenbrock_value px py);
  Printf.printf "gradient (forward mode): [%g, %g]\n" gx gy;
  Printf.printf "gradient (reverse mode): [%g, %g]\n" (adjoint grad vx) (adjoint grad vy);

  print_endline "\n=================== Jacobian of a vector-valued function ===================";
  (* f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair. *)
  let vector_fn x y =
    [ Dual.multiply x x; Dual.multiply x y; Dual.multiply (Dual.multiply y y) y ]
  in
  let column sx sy =
    List.map Dual.derivative
      (vector_fn (Dual.create ~derivative:sx 2.0) (Dual.create ~derivative:sy 3.0))
  in
  let jac = List.map2 (fun dx dy -> [ dx; dy ]) (column 1.0 0.0) (column 0.0 1.0) in
  print_endline "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:";
  List.iter print_row jac;

  print_endline
    "\n=================== Hessian via second-order forward mode (HyperDual) ===================";
  (* One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2. *)
  let mx = 1.0 and my = 1.0 (* the Rosenbrock function's minimum *) in
  let seeds = [ (1.0, 0.0); (0.0, 1.0) ] in
  let h =
    List.map
      (fun (e1x, e1y) ->
        List.map
          (fun (e2x, e2y) ->
            HyperDual.eps1eps2
              (rosenbrock_hyper_dual
                 (HyperDual.create ~eps1:e1x ~eps2:e2x mx)
                 (HyperDual.create ~eps1:e1y ~eps2:e2y my)))
          seeds)
      seeds
  in
  print_endline "Hessian of the Rosenbrock function at its minimum (1,1):";
  List.iter print_row h;

  print_endline
    "\n=================== Gradient descent driven by reverse-mode gradients ===================";
  let point = [| -1.2; 1.0 |] (* the classic Rosenbrock start *) in
  let learning_rate = 0.001 and n_steps = 2000 in
  for step = 0 to n_steps - 1 do
    let tape = Tape.create () in
    let vx = Var.create tape point.(0) in
    let vy = Var.create tape point.(1) in
    let vf = rosenbrock_var tape vx vy in
    let grad = Tape.backward tape vf in
    let loss = Var.value vf in
    point.(0) <- point.(0) -. (learning_rate *. adjoint grad vx);
    point.(1) <- point.(1) -. (learning_rate *. adjoint grad vy);
    if step = 0 || step = n_steps - 1 then
      Printf.printf "step %d: loss = %g, x = [%g, %g]\n" step loss point.(0) point.(1)
  done;
  print_endline "(true minimum is at [1, 1] with loss 0)"
