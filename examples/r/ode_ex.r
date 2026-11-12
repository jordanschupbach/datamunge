# CMD: nix develop .#r --command Rscript examples/r/ode_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
# solve_builtin() integrates a fixed set of named systems; solve() takes a custom right-hand
# side, written as an R6 subclass of RHS.
library(datamunger)

ExponentialDecay <- R6::R6Class("ExponentialDecay", inherit = RHS, public = list(
  k = NULL,
  initialize = function(k) { super$initialize(); self$k <- k },
  evaluate = function(t, y) -self$k * y
))

options <- ODEOptions$new()
options$method <- StepMethod$RK4
options$step_size <- 0.01
solver <- ODESolver$new(options)

last_state <- function(sol) sol$state_at(sol$size() - 1)

cat("=================== dy/dt = -y, y(0) = 1 (custom R RHS) ===================\n")
sol <- solver$solve(ExponentialDecay$new(1.0), c(1.0), 0.0, 2.0)
cat("RK4 with an R6 RHS: y(2) =", last_state(sol)[1], " exact:", exp(-2.0),
    " function evaluations:", sol$function_evaluations, "\n\n")

cat("=================== dy/dt = -y, y(0) = 1 ===================\n")
exact <- exp(-2.0)
cat("exact y(2) =", exact, "\n")

for (method_name in c("RK4", "RK45", "AdamsMoulton")) {
  opts <- ODEOptions$new()
  if (method_name == "RK4") {
    opts$method <- StepMethod$RK4
    opts$step_size <- 0.01
  } else if (method_name == "RK45") {
    opts$method <- StepMethod$RK45
    opts$step_size <- 0.1
    opts$abs_tol <- 1e-8
    opts$rel_tol <- 1e-8
  } else {
    opts$method <- StepMethod$AdamsMoulton
    opts$multistep_order <- 4
    opts$step_size <- 0.01
  }
  s <- ODESolver$new(opts)
  sol <- s$solve_builtin("exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  y_final <- last_state(sol)
  cat(sprintf("%-14s y(2) = %.6f  steps = %d\n", method_name, y_final[1], sol$steps_taken))
}

cat("\n=================== Harmonic oscillator (energy conservation) ===================\n")
sol <- solver$solve_builtin("harmonic_oscillator", c(1.0), c(1.0, 0.0), 0.0, 20.0)
final <- last_state(sol)
cat("x(20) =", final[1], " (cos(20) =", cos(20), ")\n")
cat("energy x^2+v^2 =", final[1]^2 + final[2]^2, "(should stay near 1.0)\n")

n <- sol$size()
t_vals <- sapply(0:(n - 1), function(i) sol$time_at(i))
x_vals <- sapply(0:(n - 1), function(i) sol$state_at(i)[1])
v_vals <- sapply(0:(n - 1), function(i) sol$state_at(i)[2])

plot_obj <- RPlot$plot(t_vals, x_vals, "l", "x(t)")
plot_obj <- plot_obj$lines(t_vals, v_vals, "v(t)")
plot_obj$title("Harmonic Oscillator")
plot_obj$x_label("t")
plot_obj$y_label("state")
plot_obj$save_svg("ode_harmonic_oscillator.svg")
cat("wrote ode_harmonic_oscillator.svg\n")

cat("\n=================== Lorenz attractor (phase plane) ===================\n")
lorenz_options <- ODEOptions$new()
lorenz_options$method <- StepMethod$RK4
lorenz_options$step_size <- 0.005
lorenz_solver <- ODESolver$new(lorenz_options)
lorenz_sol <- lorenz_solver$solve_builtin("lorenz", c(10.0, 28.0, 8.0 / 3.0), c(1.0, 1.0, 1.0), 0.0, 25.0)
cat("steps_taken =", lorenz_sol$steps_taken, "\n")

nl <- lorenz_sol$size()
xs <- sapply(0:(nl - 1), function(i) lorenz_sol$state_at(i)[1])
zs <- sapply(0:(nl - 1), function(i) lorenz_sol$state_at(i)[3])
lorenz_plot <- RPlot$plot(xs, zs, "l", "trajectory")
lorenz_plot$title("Lorenz Attractor (x-z phase plane)")
lorenz_plot$x_label("x")
lorenz_plot$y_label("z")
lorenz_plot$save_svg("ode_lorenz_phase_plane.svg")
cat("wrote ode_lorenz_phase_plane.svg\n")
