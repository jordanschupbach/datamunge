# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/optim_advanced_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Objective functions are R6 subclasses of the native function-type interfaces
# (ArbitraryFunction, DifferentiableFunction, HessianFunction, ResidualFunction, ...): the
# optimizers call the R methods back from C++.
library(datamunger)

# A 2-D quadratic bowl f(x) = (x - target)^T A (x - target) with a known minimum, exposed with
# its exact (constant) Hessian -- the natural example for Newton's method.
QuadraticBowlHessian <- R6::R6Class("QuadraticBowlHessian", inherit = HessianFunction, public = list(
  target = NULL,
  # SPD matrix, so the bowl is well-conditioned and Newton converges in one step.
  A = matrix(c(4.0, 0.5, 0.5, 2.0), 2, 2),
  initialize = function(target) {
    super$initialize()
    self$target <- target
  },
  evaluate = function(x) {
    d <- x - self$target
    drop(t(d) %*% self$A %*% d)
  },
  gradient = function(x) drop(2.0 * self$A %*% (x - self$target)),
  hessian = function(x) lapply(1:2, function(i) 2.0 * self$A[i, ])
))

# Residuals for fitting A*exp(-k*t) + c to synthetic noise-free data -- the classic nonlinear
# least-squares curve fit, and the textbook use case for LevenbergMarquardt.
ExpDecayResidual <- R6::R6Class("ExpDecayResidual", inherit = ResidualFunction, public = list(
  ts = NULL, ys = NULL,
  initialize = function(ts, ys) {
    super$initialize()
    self$ts <- ts
    self$ys <- ys
  },
  residuals = function(p) p[1] * exp(-p[2] * self$ts) + p[3] - self$ys,
  jacobian = function(p) {
    e <- exp(-p[2] * self$ts)
    lapply(seq_along(self$ts), function(i) c(e[i], -p[1] * self$ts[i] * e[i], 1.0))
  }
))

# optimize() updates its coordinates in place: pass a rosetta_ref() and read $value afterwards.
cat("=================== HessianFunction: Newton's method on a quadratic bowl ===================\n")
target <- c(3.0, -1.5)
x <- rosetta_ref(c(0.0, 0.0))
newton_options <- NewtonOptions$new()
newton_options$max_iterations <- 50
newton_options$tolerance <- 1e-10
value <- Newton$new(newton_options)$optimize(QuadraticBowlHessian$new(target), x)
cat("Newton:            f=", value, " x=", x$value, " (true minimum: f=0 at", target, ")\n")

cat("\n=================== HessianFunction: TrustRegionNewton on the same bowl ===================\n")
x <- rosetta_ref(c(5.0, 5.0))
trn_options <- TrustRegionNewtonOptions$new()
trn_options$max_iterations <- 100
trn_options$tolerance <- 1e-10
value <- TrustRegionNewton$new(trn_options)$optimize(QuadraticBowlHessian$new(target), x)
cat("TrustRegionNewton: f=", value, " x=", x$value, " (true minimum: f=0 at", target, ")\n")

cat("\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================\n")
true_params <- c(5.0, 0.7, 1.0)
ts <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ys <- true_params[1] * exp(-true_params[2] * ts) + true_params[3]
p <- rosetta_ref(c(1.0, 0.1, 0.0))
lm_options <- LevenbergMarquardtOptions$new()
lm_options$max_iterations <- 200
lm_options$tolerance <- 1e-12
value <- LevenbergMarquardt$new(lm_options)$optimize(ExpDecayResidual$new(ts, ys), p)
cat("LevenbergMarquardt: sum-sq-residual=", value, " params(A,k,c)=", p$value, "\n")
cat("                    true params(A,k,c)=", true_params, "\n")

cat("\n=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================\n")
# A 1-D bowl with a known minimum near x=1.7 and no gradient -- the black-box setting
# BayesianOptimization targets.
Shifted1DBowl <- R6::R6Class("Shifted1DBowl", inherit = ArbitraryFunction, public = list(
  evaluate = function(x) (x[1] - 1.7)^2 + 0.1 * sin(10.0 * x[1])
))
x <- rosetta_ref(c(0.0))
surrogate <- RBFGaussianProcessSurrogate$new(1.0, 1e-6)
bo_options <- BayesianOptimizationOptions$new()
bo_options$initial_samples <- 10
bo_options$max_iterations <- 60
bo_options$seed <- 42
value <- BayesianOptimization$new(bo_options)$optimize(Shifted1DBowl$new(), x, c(-3.0), c(5.0), surrogate)
cat("BayesianOptimization: f=", value, " x=", x$value, " (target minimum near x=1.7)\n")
