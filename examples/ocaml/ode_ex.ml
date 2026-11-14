(* CMD: just TARGET=ode_ex run-ocaml *)
(* ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot. *)
open Datamunge

let last_state sol = ODESolution.state_at sol (ODESolution.size sol - 1)

(* The time vector of a solution, one value per step. *)
let times sol = Array.init (ODESolution.size sol) (ODESolution.time_at sol)

(* State component [c] (0-based) of a solution, one value per step. *)
let component sol c =
  Array.init (ODESolution.size sol) (fun i -> (ODESolution.state_at sol i).(c))

let rk4_solver step =
  let options = ODEOptions.create () in
  ODEOptions.set_method_ options StepMethod.RK4;
  ODEOptions.set_step_size options step;
  ODESolver.create ~options ()

(* NOTE: like the Lua version this example uses the built-in named systems via solve_builtin. *)

let lorenz_params = [| 10.0; 28.0; 8.0 /. 3. |]

let () =
  print_endline
    "=================== Every named built-in system (no director needed) ===================";
  let solver = ODESolver.create () in
  List.iter
    (fun (name, params, y0) ->
      let sol = ODESolver.solve_builtin solver name params y0 0.0 1.0 in
      let state =
        String.concat ", " (Array.to_list (Array.map (Printf.sprintf "%g") (last_state sol)))
      in
      Printf.printf "%-22s steps=%-6d final state=[%s]\n" name (ODESolution.steps_taken sol) state)
    [
      ("exponential_decay", [| 1.0 |], [| 1.0 |]);
      ("logistic_growth", [| 1.0; 1.0 |], [| 0.5 |]);
      ("harmonic_oscillator", [| 1.0 |], [| 1.0; 0.0 |]);
      ("van_der_pol", [| 1.0 |], [| 2.0; 0.0 |]);
      ("lorenz", lorenz_params, [| 1.0; 1.0; 1.0 |]);
    ];

  print_endline
    "\n=================== Harmonic oscillator (energy conservation) ===================";
  let sol =
    ODESolver.solve_builtin (rk4_solver 0.01) "harmonic_oscillator" [| 1.0 |] [| 1.0; 0.0 |] 0.0
      20.0
  in
  let final = last_state sol in
  let x = final.(0) and v = final.(1) in
  Printf.printf "x(20) = %.6f (cos(20) = %.6f)\n" x (cos 20.0);
  Printf.printf "energy x^2+v^2 = %.6f (should stay near 1.0)\n" ((x *. x) +. (v *. v));
  let ts = times sol in
  let p = RPlot.plot ~type_:"l" ~label:"x(t)" ts (component sol 0) in
  ignore (RPlot.lines ~label:"v(t)" p ts (component sol 1));
  let p = RPlot.y_label (RPlot.x_label (RPlot.title p "Harmonic Oscillator") "t") "state" in
  RPlot.save_svg p "ode_harmonic_oscillator_ocaml.svg";
  print_endline "wrote ode_harmonic_oscillator_ocaml.svg";

  print_endline "\n=================== Lorenz attractor (phase plane) ===================";
  let sol =
    ODESolver.solve_builtin (rk4_solver 0.005) "lorenz" lorenz_params [| 1.0; 1.0; 1.0 |] 0.0 25.0
  in
  Printf.printf "steps_taken = %d\n" (ODESolution.steps_taken sol);
  let p = RPlot.plot ~type_:"l" ~label:"trajectory" (component sol 0) (component sol 2) in
  let p =
    RPlot.y_label (RPlot.x_label (RPlot.title p "Lorenz Attractor (x-z phase plane)") "x") "z"
  in
  RPlot.save_svg p "ode_lorenz_phase_plane_ocaml.svg";
  print_endline "wrote ode_lorenz_phase_plane_ocaml.svg"
