test_that("ODESolver's RK4 matches exponential decay's closed form", {
  options <- ODEOptions()
  options$method <- StepMethod_StepMethod_RK4_get()
  options$step_size <- 0.01
  solver <- ODESolver__SWIG_0(options)

  sol <- ODESolver_solve_builtin(solver, "exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- ODESolution_state_at(sol, ODESolution_size(sol) - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
  expect_equal(ODESolution_time_at(sol, 0), 0.0)
  expect_equal(ODESolution_time_at(sol, ODESolution_size(sol) - 1), 2.0, tolerance = 1e-9)
})

test_that("ODESolver's adaptive RK45 matches exponential decay and uses few steps", {
  options <- ODEOptions()
  options$method <- StepMethod_StepMethod_RK45_get()
  options$step_size <- 0.1
  options$abs_tol <- 1e-8
  options$rel_tol <- 1e-8
  solver <- ODESolver__SWIG_0(options)

  sol <- ODESolver_solve_builtin(solver, "exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- ODESolution_state_at(sol, ODESolution_size(sol) - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
  expect_lt(ODESolution_steps_taken_get(sol), 50)
})

test_that("ODESolver's Adams-Bashforth-Moulton matches exponential decay", {
  options <- ODEOptions()
  options$method <- StepMethod_StepMethod_AdamsMoulton_get()
  options$multistep_order <- 4
  options$step_size <- 0.001
  solver <- ODESolver__SWIG_0(options)

  sol <- ODESolver_solve_builtin(solver, "exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- ODESolution_state_at(sol, ODESolution_size(sol) - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
})

test_that("ODESolver's harmonic_oscillator matches the closed-form solution", {
  options <- ODEOptions()
  options$method <- StepMethod_StepMethod_RK4_get()
  options$step_size <- 0.001
  solver <- ODESolver__SWIG_0(options)

  sol <- ODESolver_solve_builtin(solver, "harmonic_oscillator", c(1.0), c(1.0, 0.0), 0.0, 10.0)
  last <- ODESolution_state_at(sol, ODESolution_size(sol) - 1)
  expect_equal(last[1], cos(10), tolerance = 1e-4)
  expect_equal(last[2], -sin(10), tolerance = 1e-4)
})

test_that("ODESolver covers every named builtin system and supports $ dispatch", {
  solver <- ODESolver__SWIG_1()

  sol1 <- ODESolver_solve_builtin(solver, "logistic_growth", c(1.0, 1.0), c(0.5), 0.0, 1.0)
  expect_equal(ODESolution_size(sol1) > 0, TRUE)

  sol2 <- ODESolver_solve_builtin(solver, "van_der_pol", c(1.0), c(2.0, 0.0), 0.0, 1.0)
  expect_equal(length(ODESolution_state_at(sol2, 0)), 2)

  sol3 <- ODESolver_solve_builtin(solver, "lorenz", numeric(0), c(1.0, 1.0, 1.0), 0.0, 1.0)
  expect_equal(length(ODESolution_state_at(sol3, 0)), 3)

  expect_equal(sol3$size(), ODESolution_size(sol3))
  expect_equal(sol3$state_at(0), ODESolution_state_at(sol3, 0))
})
