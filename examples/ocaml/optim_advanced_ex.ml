(* CMD: just TARGET=optim_advanced_ex run-ocaml *)
(* Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from OCaml. *)
open Datamunge

(* Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
   EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
   ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
   implement the objective. From OCaml, [extend] the interface class with closures for its
   virtual methods; the optimizer then calls the OCaml closures. The coordinates are passed as a
   [float array ref] and updated in place. *)

(* f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian. *)
let quadratic () =
  HessianFunction.extend
    ~evaluate:(fun _ c -> (((c.(0) -. 1.) ** 2.) +. (10. *. ((c.(1) +. 2.) ** 2.))))
    ~gradient:(fun _ c -> [| 2. *. (c.(0) -. 1.); 20. *. (c.(1) +. 2.) |])
    ~hessian:(fun _ _ -> [ [| 2.0; 0.0 |]; [| 0.0; 20.0 |] ])
    ()

(* Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5. *)
let exponential ts =
  ResidualFunction.extend
    ~residuals:(fun _ p ->
      Array.map (fun t -> (p.(0) *. exp (p.(1) *. t)) -. (2. *. exp (0.5 *. t))) ts)
    ~jacobian:(fun _ p ->
      Array.to_list
        (Array.map (fun t -> [| exp (p.(1) *. t); p.(0) *. t *. exp (p.(1) *. t) |]) ts))
    ()

(* Leaves gradient and hessian out. *)
let incomplete () = HessianFunction.extend ~evaluate:(fun _ _ -> 0.0) ()

(* Whether [needle] occurs in [s]. *)
let contains s needle =
  let n = String.length needle in
  let rec at i = i + n <= String.length s && (String.sub s i n = needle || at (i + 1)) in
  at 0

(* Reports that an optimizer of the given class was constructed. *)
let constructed name (_ : _ obj) = Printf.printf "%s: %s()\n" name name

let () =
  print_endline
    "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) \
     ===================";
  (* Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
     then check the acquisition function favors an unexplored region. *)
  let xs = [| 0.0; 0.5; 1.0; 1.5; 2.0; 3.0 |] in
  let ys = Array.map (fun x -> (x -. 1.) ** 2.) xs in
  let surrogate = RBFGaussianProcessSurrogate.create ~length_scale:1.0 ~noise:1e-6 () in
  RBFGaussianProcessSurrogate.fit surrogate (Array.to_list (Array.map (fun x -> [| x |]) xs)) ys;
  let incumbent = Array.fold_left Float.min 0.0 ys (* best observed value so far *) in
  Printf.printf "incumbent (best observed y): %g\n" incumbent;
  List.iter
    (fun x ->
      Printf.printf "acquisition(x=%.2f) = %.6f\n" x
        (RBFGaussianProcessSurrogate.acquisition surrogate [| x |] incumbent))
    [ 0.25; 1.0; 2.5 ];

  print_endline
    "\n=================== Options structs + optimizer construction (all 10 new classes) \
     ===================";
  (* Every new Options struct and optimizer class constructs cleanly, proving
     the binding for the whole batch is wired correctly. *)
  let pg_options = ProximalGradientOptions.create () in
  ProximalGradientOptions.set_step_size pg_options 0.05;
  constructed "ProximalGradient" (ProximalGradient.create ~options:pg_options ());
  constructed "FISTA" (FISTA.create ~options:(FISTAOptions.create ()) ());
  let lm_options = LevenbergMarquardtOptions.create () in
  LevenbergMarquardtOptions.set_max_iterations lm_options 200;
  let lm = LevenbergMarquardt.create ~options:lm_options () in
  constructed "LevenbergMarquardt" lm;
  let newton = Newton.create ~options:(NewtonOptions.create ()) () in
  constructed "Newton" newton;
  constructed "TrustRegionNewton"
    (TrustRegionNewton.create ~options:(TrustRegionNewtonOptions.create ()) ());
  constructed "AugmentedLagrangian"
    (AugmentedLagrangian.create ~options:(AugmentedLagrangianOptions.create ()) ());
  constructed "SQP" (SQP.create ~options:(SQPOptions.create ()) ());
  constructed "InteriorPoint" (InteriorPoint.create ~options:(InteriorPointOptions.create ()) ());
  let bo_options = BayesianOptimizationOptions.create () in
  BayesianOptimizationOptions.set_initial_samples bo_options 8;
  constructed "BayesianOptimization" (BayesianOptimization.create ~options:bo_options ());

  print_endline
    "\n=================== OCaml objectives (subclassing the C++ interfaces) ===================";
  (* Newton's method on the quadratic bowl. *)
  let start = ref [| 5.0; 5.0 |] in
  let value = Newton.optimize newton (quadratic ()) start in
  Printf.printf "Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n" value !start.(0)
    !start.(1);

  (* Levenberg-Marquardt fitting y = a * exp(b t). *)
  let params = ref [| 1.0; 0.1 |] in
  let sse = LevenbergMarquardt.optimize lm (exponential [| 0.0; 0.5; 1.0; 1.5; 2.0 |]) params in
  Printf.printf "LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)\n" sse
    !params.(0) !params.(1);

  (* An interface method a subclass leaves out raises when the optimizer calls it. *)
  let ok, mentions =
    match Newton.optimize newton (incomplete ()) (ref [| 0.0; 0.0 |]) with
    | _ -> (true, false)
    | exception Error (_, msg) -> (false, contains msg "gradient")
  in
  Printf.printf "missing gradient(): ok=%b, error mentions it: %b\n" ok mentions
