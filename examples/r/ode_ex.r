# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# ODESolver has no facade DataFrame dependency, so it's %include-d directly rather than wrapped
# -- see ode_ex.cpp for the C++-native custom-RHS path; R has no SWIG director support at all
# (project-wide, not specific to this class), so R can only use solve_builtin()'s fixed set of
# named systems, not a live user-supplied right-hand side.
library(datamunger)

options <- ODEOptions()
options$method <- StepMethod_StepMethod_RK4_get()
options$step_size <- 0.01
solver <- ODESolver__SWIG_0(options)

last_state <- function(sol) ODESolution_state_at(sol, ODESolution_size(sol) - 1)

cat("=================== dy/dt = -y, y(0) = 1 ===================\n")
exact <- exp(-2.0)
cat("exact y(2) =", exact, "\n")

for (method_name in c("RK4", "RK45", "AdamsMoulton")) {
  opts <- ODEOptions()
  if (method_name == "RK4") {
    opts$method <- StepMethod_StepMethod_RK4_get()
    opts$step_size <- 0.01
  } else if (method_name == "RK45") {
    opts$method <- StepMethod_StepMethod_RK45_get()
    opts$step_size <- 0.1
    opts$abs_tol <- 1e-8
    opts$rel_tol <- 1e-8
  } else {
    opts$method <- StepMethod_StepMethod_AdamsMoulton_get()
    opts$multistep_order <- 4
    opts$step_size <- 0.01
  }
  s <- ODESolver__SWIG_0(opts)
  sol <- ODESolver_solve_builtin(s, "exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  y_final <- last_state(sol)
  cat(sprintf("%-14s y(2) = %.6f  steps = %d\n", method_name, y_final[1], ODESolution_steps_taken_get(sol)))
}

cat("\n=================== Harmonic oscillator (energy conservation) ===================\n")
sol <- ODESolver_solve_builtin(solver, "harmonic_oscillator", c(1.0), c(1.0, 0.0), 0.0, 20.0)
final <- last_state(sol)
cat("x(20) =", final[1], " (cos(20) =", cos(20), ")\n")
cat("energy x^2+v^2 =", final[1]^2 + final[2]^2, "(should stay near 1.0)\n")

n <- ODESolution_size(sol)
t_vals <- sapply(0:(n - 1), function(i) ODESolution_time_at(sol, i))
x_vals <- sapply(0:(n - 1), function(i) ODESolution_state_at(sol, i)[1])
v_vals <- sapply(0:(n - 1), function(i) ODESolution_state_at(sol, i)[2])

plot_obj <- RPlot_plot__SWIG_2(t_vals, x_vals, "l", "x(t)")
plot_obj <- RPlot_lines__SWIG_2(plot_obj, t_vals, v_vals, "v(t)")
Plot_title(plot_obj, "Harmonic Oscillator")
Plot_x_label(plot_obj, "t")
Plot_y_label(plot_obj, "state")
Plot_save_svg(plot_obj, "ode_harmonic_oscillator.svg")
cat("wrote ode_harmonic_oscillator.svg\n")

cat("\n=================== Lorenz attractor (phase plane) ===================\n")
lorenz_options <- ODEOptions()
lorenz_options$method <- StepMethod_StepMethod_RK4_get()
lorenz_options$step_size <- 0.005
lorenz_solver <- ODESolver__SWIG_0(lorenz_options)
lorenz_sol <- ODESolver_solve_builtin(lorenz_solver, "lorenz", c(10.0, 28.0, 8.0 / 3.0), c(1.0, 1.0, 1.0), 0.0, 25.0)
cat("steps_taken =", ODESolution_steps_taken_get(lorenz_sol), "\n")

nl <- ODESolution_size(lorenz_sol)
xs <- sapply(0:(nl - 1), function(i) ODESolution_state_at(lorenz_sol, i)[1])
zs <- sapply(0:(nl - 1), function(i) ODESolution_state_at(lorenz_sol, i)[3])
lorenz_plot <- RPlot_plot__SWIG_2(xs, zs, "l", "trajectory")
Plot_title(lorenz_plot, "Lorenz Attractor (x-z phase plane)")
Plot_x_label(lorenz_plot, "x")
Plot_y_label(lorenz_plot, "z")
Plot_save_svg(lorenz_plot, "ode_lorenz_phase_plane.svg")
cat("wrote ode_lorenz_phase_plane.svg\n")
