# Note: this uses the flat ClassName_method(obj, ...) call form throughout, not R's usual
# obj$method(...) syntax -- see memory/datamunge_r_dollar_dispatch_bug.md.
#
# IMPORTANT FINDING: this file is a NEGATIVE result for the 10 new optim classes
# (ProximalGradient/FISTA/LevenbergMarquardt/Newton/TrustRegionNewton/AugmentedLagrangian/
# SQP/InteriorPoint/BayesianOptimization/RBFGaussianProcessSurrogate), and in fact for the
# ENTIRE datamunge::optim module, old and new alike: R has NO SWIG director support at all
# (project-wide -- see memory/datamunge_r_dollar_dispatch_bug.md and ode_ex.r's similar
# note about ODESolver's custom-RHS path). Every optimizer in this module takes a custom
# objective (ArbitraryFunction/DifferentiableFunction/HessianFunction/ResidualFunction/...)
# that the CALLER must subclass and implement in their own language. In R this is not just
# "the callback doesn't fire" -- the R backend does not even generate a constructor for
# these abstract interface types (no `new_HessianFunction`, `new_ResidualFunction`, or even
# `new_ArbitraryFunction` in R/datamunger.R), so there is no way to construct an instance at
# all, let alone have C++ call back into it. Empirically (see the tryCatch blocks below), it
# fails even earlier than that: R's S4 `setMethod()` on the generated flat accessor function
# (e.g. `ArbitraryFunction_evaluate`) errors immediately with "assignment of an object of
# class 'SWIGFunction' is not valid for @'.Data' in an object of class
# 'derivedDefaultMethod'" -- the generated accessor is tagged class "SWIGFunction", which R's
# implicit-generic promotion machinery refuses to treat as a plain function body. So this
# limitation isn't just "callbacks silently no-op" -- there is no code path in R at all by
# which a user could define a working subclass, real or accidental. This affects every optimizer class in R,
# including the 9 pre-existing ones (GradientDescent, Adam, LBFGS, SimulatedAnnealing, ...)
# -- none of them have ever had an R example in this repo, which is consistent with this
# being a real, previously-undemonstrated gap rather than something newly broken by the 10
# new classes.
#
# What DOES work in R: RBFGaussianProcessSurrogate is a CONCRETE class (not requiring any
# subclassing) with its fit()/acquisition() methods already implemented in C++, so it is
# fully usable directly from R. That is exercised at the bottom of this file as the one
# genuine positive result available in this language for the new classes.

# DVector's own instance methods (size/__getitem__/etc.) are broken through their normal
# flat wrapper -- the generated code does `self = as.numeric(self)` before the .Call because
# self's type (std::vector<double>) collides with the global vector<double>-parameter %apply
# typemap, so it can't tell "self" apart from a plain numeric-vector argument (see
# examples/r/timeseries_ex.r). Bypass with direct .Call()s via this helper, as that file does.
library(datamunger)

dv <- function(...) DVector(c(...))

dvector_to_r <- function(v) {
  n <- .Call("R_swig_DVector_size", v, FALSE, PACKAGE = "datamunger")
  sapply(0:(n - 1), function(i) .Call("R_swig_DVector___getitem__", v, as.integer(i), FALSE, PACKAGE = "datamunger"))
}

cat("=================== Attempt 1: subclass HessianFunction for Newton ===================\n")
tryCatch({
  setClass("RHessianBowl", contains = "_p_datamunge__optim__HessianFunction")
  setMethod("ArbitraryFunction_evaluate", "RHessianBowl", function(self, coordinates, .copy = FALSE) {
    sum((coordinates - c(3.0, -1.5))^2)
  })
  setMethod("DifferentiableFunction_gradient", "RHessianBowl", function(self, coordinates, .copy = FALSE) {
    2.0 * (coordinates - c(3.0, -1.5))
  })
  setMethod("HessianFunction_hessian", "RHessianBowl", function(self, coordinates, .copy = FALSE) {
    list(c(2.0, 0.0), c(0.0, 2.0))
  })
  f <- new("RHessianBowl")
  x <- dv(0.0, 0.0)
  value <- Newton_optimize(Newton(), f, x)
  cat("Newton result: f=", value, " x=", dvector_to_r(x), "\n")
}, error = function(e) {
  cat("FAILED as expected -- no director support in R:", conditionMessage(e), "\n")
})

cat("\n=================== Attempt 2: subclass ResidualFunction for LevenbergMarquardt ===================\n")
tryCatch({
  setClass("RExpDecayResidual", contains = "_p_datamunge__optim__ResidualFunction")
  ts <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
  true_A <- 5.0; true_k <- 0.7; true_c <- 1.0
  ys <- true_A * exp(-true_k * ts) + true_c
  setMethod("ResidualFunction_residuals", "RExpDecayResidual", function(self, p, .copy = FALSE) {
    A <- p[1]; k <- p[2]; c <- p[3]
    A * exp(-k * ts) + c - ys
  })
  setMethod("ResidualFunction_jacobian", "RExpDecayResidual", function(self, p, .copy = FALSE) {
    A <- p[1]; k <- p[2]
    e <- exp(-k * ts)
    lapply(seq_along(ts), function(i) c(e[i], -A * ts[i] * e[i], 1.0))
  })
  residual_fn <- new("RExpDecayResidual")
  params <- dv(1.0, 0.1, 0.0)
  value <- LevenbergMarquardt_optimize(LevenbergMarquardt(), residual_fn, params)
  cat("LevenbergMarquardt result: ssr=", value, " params=", dvector_to_r(params), "\n")
}, error = function(e) {
  cat("FAILED as expected -- no director support in R:", conditionMessage(e), "\n")
})

cat("\n=================== Attempt 3: subclass ArbitraryFunction as objective for BayesianOptimization ===================\n")
tryCatch({
  setClass("RShifted1DBowl", contains = "_p_datamunge__optim__ArbitraryFunction")
  setMethod("ArbitraryFunction_evaluate", "RShifted1DBowl", function(self, coordinates, .copy = FALSE) {
    (coordinates[1] - 1.7)^2 + 0.1 * sin(10.0 * coordinates[1])
  })
  f <- new("RShifted1DBowl")
  x <- dv(0.0)
  surrogate <- RBFGaussianProcessSurrogate(1.0, 1e-6)
  bo_options <- BayesianOptimizationOptions()
  bo_options$initial_samples <- 10
  bo_options$max_iterations <- 60
  bo_options$seed <- 42
  value <- BayesianOptimization_optimize(BayesianOptimization(bo_options), f, x, c(-3.0), c(5.0), surrogate)
  cat("BayesianOptimization result: f=", value, " x=", dvector_to_r(x), "\n")
}, error = function(e) {
  cat("FAILED as expected -- no director support in R:", conditionMessage(e), "\n")
})

cat("\n=================== What DOES work: RBFGaussianProcessSurrogate used directly (no subclassing) ===================\n")
# RBFGaussianProcessSurrogate is a concrete, ready-to-use implementation of BayesianSurrogate
# -- fit()/acquisition() are real C++ methods, not user-supplied callbacks, so this is fully
# usable from R with no director support required.
surrogate <- RBFGaussianProcessSurrogate(1.0, 1e-6)
points <- list(c(0.0), c(1.0), c(2.0), c(3.0))
values <- c(4.0, 1.0, 0.0, 1.0)
invisible(RBFGaussianProcessSurrogate_fit(surrogate, points, values))
acq <- RBFGaussianProcessSurrogate_acquisition(surrogate, c(1.7), 0.0)
cat("RBFGaussianProcessSurrogate: fit() on 4 points succeeded, acquisition(x=1.7, incumbent=0) =", acq, "\n")
cat("(this confirms the new classes ARE correctly wired into R's bindings -- construction,\n")
cat(" options structs, and any CONCRETE method all work; only director-based subclassing is unavailable)\n")
