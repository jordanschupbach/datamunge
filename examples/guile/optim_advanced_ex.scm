(use-modules (datamunge))
(use-modules (ice-9 format))

;; This example exercises the 10 new datamunge::optim classes (ProximalGradient, FISTA,
;; LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint,
;; BayesianOptimization, RBFGaussianProcessSurrogate) that were just wired into
;; src/datamungeguile/swig/datamungeguile.i.
;;
;; IMPORTANT FINDING, read before assuming this example is "supposed to" run an optimizer
;; end-to-end the way the Python/Ruby/etc. optim_ex does: every one of these 10 classes
;; takes a user-supplied objective as a director-callback interface type (ProximalFunction,
;; HessianFunction, EqualityConstrainedFunction, InequalityConstrainedFunction,
;; ResidualFunction, or ArbitraryFunction+BayesianSurrogate). Guile's SWIG backend, as built
;; by this project's justfile (`swig -guile -c++ ...`, no `-proxy` flag), generates NO
;; director/subclass mechanism at all -- confirmed empirically below and independently by
;; `grep -c -i director build/datamungeguile-swig/datamunge_guile_wrap.cxx` returning 0 even
;; though datamungeguile.i has `%feature("director")` on all of these interfaces (including
;; the pre-existing ArbitraryFunction/DifferentiableFunction/etc.). This matches this
;; project's own prior recorded finding ("Guile has zero director support"; see
;; datamunge_guile_bindings memory) -- it is NOT stale, and it is NOT specific to these 10
;; new classes: it affects the entire datamunge::optim and datamunge::bayes modules, every
;; class that has ever required a user-defined objective, going all the way back to
;; GradientDescent/Adam/SimulatedAnnealing/etc. None of those have a Guile example either,
;; for the same reason.
;;
;; So this example demonstrates what genuinely works from Guile today: constructing every
;; new Options struct and every new optimizer object (proves the SWIG wiring/codegen for
;; these 10 classes is otherwise sound), constructing the one concrete/ready-to-use class
;; (RBFGaussianProcessSurrogate), and then an explicit, honest attempt at the director/
;; subclass test the verification brief calls for -- showing exactly how and why it fails.

(format #t "=== Part 1: construction sanity-check for all 10 new classes ===\n\n")

(define (check-options name ctor)
  (let ((opts (ctor)))
    (format #t "~a: constructed OK (~a)\n" name opts)
    opts))

(define proximal-opts (check-options "ProximalGradientOptions" new-ProximalGradientOptions))
(ProximalGradientOptions-max-iterations-set proximal-opts 5000)
(define proximal-gradient (new-ProximalGradient proximal-opts))
(format #t "ProximalGradient optimizer object: ~a\n\n" proximal-gradient)

(define fista-opts (check-options "FISTAOptions" new-FISTAOptions))
(define fista (new-FISTA fista-opts))
(format #t "FISTA optimizer object: ~a\n\n" fista)

(define lm-opts (check-options "LevenbergMarquardtOptions" new-LevenbergMarquardtOptions))
(LevenbergMarquardtOptions-max-iterations-set lm-opts 200)
(define lm (new-LevenbergMarquardt lm-opts))
(format #t "LevenbergMarquardt optimizer object: ~a\n\n" lm)

(define newton-opts (check-options "NewtonOptions" new-NewtonOptions))
(define newton (new-Newton newton-opts))
(format #t "Newton optimizer object: ~a\n\n" newton)

(define trn-opts (check-options "TrustRegionNewtonOptions" new-TrustRegionNewtonOptions))
(define trust-region-newton (new-TrustRegionNewton trn-opts))
(format #t "TrustRegionNewton optimizer object: ~a\n\n" trust-region-newton)

(define al-opts (check-options "AugmentedLagrangianOptions" new-AugmentedLagrangianOptions))
(define augmented-lagrangian (new-AugmentedLagrangian al-opts))
(format #t "AugmentedLagrangian optimizer object: ~a\n\n" augmented-lagrangian)

(define sqp-opts (check-options "SQPOptions" new-SQPOptions))
(define sqp (new-SQP sqp-opts))
(format #t "SQP optimizer object: ~a\n\n" sqp)

(define ip-opts (check-options "InteriorPointOptions" new-InteriorPointOptions))
(define interior-point (new-InteriorPoint ip-opts))
(format #t "InteriorPoint optimizer object: ~a\n\n" interior-point)

(define bo-opts (check-options "BayesianOptimizationOptions" new-BayesianOptimizationOptions))
(define bayesian-optimization (new-BayesianOptimization bo-opts))
(format #t "BayesianOptimization optimizer object: ~a\n\n" bayesian-optimization)

;; RBFGaussianProcessSurrogate is the one class of the 10 that is concrete/ready-to-use --
;; it implements BayesianSurrogate itself in C++, so no Guile-side subclass is required.
(define surrogate (new-RBFGaussianProcessSurrogate 1.0 1e-6))
(format #t "RBFGaussianProcessSurrogate (concrete, ready-to-use): constructed OK (~a)\n\n" surrogate)

(format #t "All 10 new classes' Options structs and optimizer objects construct correctly --\n")
(format #t "the SWIG wiring/codegen itself (%include, %template<vector<vector<double>>>, etc.)\n")
(format #t "is sound for Guile. The gap is specifically the director/callback mechanism.\n\n")

(format #t "=== Part 2: the load-bearing director/subclass test ===\n\n")
(format #t "Every optimize() call among these 10 classes needs an actual instance of an\n")
(format #t "abstract interface type (HessianFunction, ResidualFunction, ProximalFunction,\n")
(format #t "EqualityConstrainedFunction, InequalityConstrainedFunction, or ArbitraryFunction).\n")
(format #t "In every OTHER language with working directors (Python/Ruby/Perl/D/Java/...), you\n")
(format #t "get such an instance by subclassing the interface type in that language, e.g. a\n")
(format #t "Scheme analogue of `class MyHessian(HessianFunction): ...`.\n\n")

(format #t "Checking whether Guile's binding exposes ANY way to construct one:\n")
(for-each
 (lambda (sym)
   (format #t "  ~a bound? ~a\n" sym (module-bound? (current-module) sym)))
 '(new-HessianFunction new-ResidualFunction new-ProximalFunction
   new-EqualityConstrainedFunction new-InequalityConstrainedFunction
   new-ArbitraryFunction new-DifferentiableFunction new-BayesianSurrogate))

(format #t "\nNone of these constructors exist -- HessianFunction/ResidualFunction/etc. are\n")
(format #t "pure-virtual C++ interfaces, and without director support Guile has no mechanism\n")
(format #t "(no GOOPS proxy class, no callback trampoline) to hand C++ a Scheme-defined\n")
(format #t "override of `hessian`/`residuals`/`evaluate`/etc. This is corroborated directly by\n")
(format #t "inspecting the generated wrapper: `grep -c -i director\n")
(format #t "build/datamungeguile-swig/datamunge_guile_wrap.cxx` returns 0 -- no `SwigDirector`\n")
(format #t "classes are emitted at all, despite %feature(\"director\") being declared on every\n")
(format #t "one of these interfaces (both the 4 new pure-callback ones added for this task and\n")
(format #t "the pre-existing ArbitraryFunction/DifferentiableFunction/SeparableFunction/\n")
(format #t "DifferentiableSeparableFunction). The `-proxy` SWIG flag (needed to export GOOPS\n")
(format #t "class definitions users could subclass from) is also not passed in this project's\n")
(format #t "`prebuild-guile` justfile recipe, so there is no class-like object in Scheme to\n")
(format #t "inherit from in the first place -- confirming the gap at two independent levels.\n\n")

(format #t "Concretely demonstrating the dead end: attempting to call Newton-optimize with\n")
(format #t "something that is NOT a HessianFunction (since none can be constructed) fails with\n")
(format #t "a type error, exactly as expected:\n")
(let ((coords (new-DVector)))
  (DVector-push! coords 0.0)
  (DVector-push! coords 0.0)
  (catch #t
    (lambda ()
      (Newton-optimize newton coords coords) ;; wrong types on purpose -- no valid HessianFunction exists
      (format #t "  (unexpectedly succeeded!)\n"))
    (lambda (key . args)
      (format #t "  Newton-optimize call failed as expected: ~a ~a\n" key args))))

(format #t "\n=== Conclusion ===\n")
(format #t "CONFIRMED (not stale): Guile bindings have zero working director/callback support.\n")
(format #t "This affects the entire datamunge::optim (and datamunge::bayes) module, not just\n")
(format #t "these 10 new classes -- every optimizer that needs a user-supplied objective is\n")
(format #t "unusable from Guile with a real custom objective, from GradientDescent/Adam all the\n")
(format #t "way through this task's 10 new classes. The %feature(\"director\") lines added for\n")
(format #t "the new interfaces in datamungeguile.i are silently non-functional decoration in\n")
(format #t "this backend/build configuration, same as the pre-existing ones already were.\n")
